from typing import Optional, List, Tuple

import itertools
import torch
from transformers import BertModel, BertTokenizerFast

from pero_ocr.core.layout import PageLayout

from textbite.bite import Bite
from textbite.models.feature_extraction import TextFeaturesProvider, GeometryFeaturesProvider
from textbite.geometry import best_intersecting_bbox, polygon_to_bbox, AABB, PageGeometry, dist_l2


class Graph:
    def __init__(
        self,
        id: str,
        node_features,
        from_indices,
        to_indices,
        edge_attr,
    ):
        self.graph_id = id
        self.node_features = torch.stack(node_features)  # Shape (n_nodes, n_features)
        self.edge_index = torch.tensor([from_indices, to_indices], dtype=torch.int64)  # Shape (2, n_edges)
        self.edge_attr = torch.stack(edge_attr)  # Shape (n_edges, n_features)


class JoinerGraphProvider:
    def __init__(
        self,
        tokenizer: Optional[BertTokenizerFast]=None,
        czert: Optional[BertModel]=None,
        device=None,
        ):
        self.text_features_provider = TextFeaturesProvider(tokenizer, czert, device)
        self.geometric_features_provider = GeometryFeaturesProvider()

    def get_transcriptions(self, regions: List[AABB], pagexml: PageLayout) -> List[str]:
        transcriptions = ["" for _ in regions]
        for line in pagexml.lines_iterator():
            line_bbox = polygon_to_bbox(line.polygon)

            idx = best_intersecting_bbox(line_bbox, regions)
            if idx is not None and line and line.transcription and line.transcription.strip():
                transcriptions[idx] += f"{line.transcription.strip()}\n"

        return transcriptions
    
    def create_all_edges(self, regions: List[AABB]) -> List[Tuple[int, int]]:
        edges = list(itertools.combinations(range(len(regions)), 2))
        edges = edges + [(to_idx, from_idx) for from_idx, to_idx in edges]
        return edges
    
    def create_edge_attr(
            self,
            edges: List[Tuple[int, int]],
            geometry: PageGeometry,
            czert_embeddings: List[Tuple[torch.FloatTensor, torch.FloatTensor]],
        ) -> List[torch.FloatTensor]:
        edge_attrs = []

        for from_idx, to_idx in edges:
            edge_attr = []

            from_region = geometry.regions[from_idx]
            to_region = geometry.regions[to_idx]

            x_dist = abs(from_region.center.x - to_region.center.x)
            y_dist = abs(from_region.center.y - to_region.center.y)
            dist = dist_l2(from_region.center, to_region.center)

            edge_attr.append(x_dist)
            edge_attr.append(y_dist)
            edge_attr.append(dist)

            edge_attr.append(x_dist / geometry.page_width)
            edge_attr.append(y_dist / geometry.page_height)

            distance_x = max(0, min(to_region.bbox.xmin - from_region.bbox.xmax, to_region.bbox.xmax - from_region.bbox.xmin))
            distance_y = max(0, min(to_region.bbox.ymin - from_region.bbox.ymax, to_region.bbox.ymax - from_region.bbox.ymin))

            edge_attr.append(distance_x)
            edge_attr.append(distance_y)
            edge_attr.append(distance_x / geometry.page_width)
            edge_attr.append(distance_y / geometry.page_height)

            from_pooler, from_cls = czert_embeddings[from_idx]
            to_pooler, to_cls = czert_embeddings[to_idx]

            pooler_dist = torch.cosine_similarity(from_pooler, to_pooler).item()
            cls_dist = torch.cosine_similarity(from_cls, to_cls).item()

            edge_attr.append(pooler_dist)
            edge_attr.append(cls_dist)

            if (from_region.child and from_region.child is to_region) or \
               (from_region.parent and from_region.parent is to_region) or \
               (to_region.child and to_region.child is from_region) or \
               (to_region.parent and to_region.parent is from_region):
                edge_attr.append(1.0)
            else:
                edge_attr.append(0.0)

            edge_attrs.append(torch.tensor(edge_attr, dtype=torch.float32))

        return edge_attrs

    def get_graph_from_bites(
        self,
        bites: List[Bite],
        filename: str,
        pagexml: PageLayout,
    ) -> Graph:
        all_regions = [bite.bbox for bite in bites]

        if len(all_regions) < 2:
            raise RuntimeError("To create graph from regions, at least 2 regions must exist")

        transcriptions = self.get_transcriptions(all_regions, pagexml)
        geometry = PageGeometry(regions=all_regions, pagexml=pagexml)
        czert_embeddings = []

        for t in transcriptions:
            czert_embedding = self.text_features_provider.get_czert_features(t)
            czert_embeddings.append(czert_embedding)

        node_features = self.geometric_features_provider.get_regions_features(geometry, pagexml)

        edges = self.create_all_edges(all_regions)

        edge_attr = self.create_edge_attr(edges, geometry, czert_embeddings)

        from_indices = [from_idx for from_idx, _ in edges]
        to_indices = [to_idx for _, to_idx in edges]

        graph = Graph(
            id=filename,
            node_features=node_features,
            from_indices=from_indices,
            to_indices=to_indices,
            edge_attr=edge_attr,
        )

        return graph
