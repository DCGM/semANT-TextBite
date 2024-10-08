from typing import List, Tuple, Dict

import torch

from pero_ocr.core.layout import PageLayout

from textbite.models.joiner.graph import JoinerGraphProvider, Graph
from textbite.models.joiner.model import JoinerGraphModel
from textbite.models.utils import edge_indices_to_edges, get_transitive_subsets, GraphNormalizer, get_similarities
from textbite.bite import Bite


def join_bites_by_dict(nodes_to_join: Dict[int, List[int]], bites: List[Bite]) -> List[Bite]:
    to_delete = []
    for kept_bite_idx, bites_indices in nodes_to_join.items():
        to_delete.extend(bites_indices)
        for bite_index in bites_indices:
            bites[kept_bite_idx].lines.extend(bites[bite_index].lines)

    bites_new = [bite for i, bite in enumerate(bites) if i not in to_delete]

    return bites_new


def get_joining_dict(positive_edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    subsets = get_transitive_subsets(positive_edges)

    nodes_to_join = []
    for subset in subsets:
        nodes = set()
        for edge in subset:
            nodes.update(edge)
        nodes_to_join.append(nodes)

    nodes_to_join = [list(ss) for ss in nodes_to_join]
    nodes_to_join = {s[0]: s[1:] for s in nodes_to_join}

    return nodes_to_join


def get_positive_edges_gcn(
        graph: Graph,
        joiner,
        device,
        threshold: float,
    ) -> List[Bite]:
    node_features = graph.node_features.to(device)
    edge_indices = graph.edge_index.to(device)
    edge_attrs = graph.edge_attr.to(device)

    with torch.no_grad():
        outputs = joiner(node_features, edge_indices, edge_attrs)
    similarities = get_similarities(outputs, edge_indices)
    similarities = similarities.tolist()
    positive_edge_indices = [index for index, similarity in enumerate(similarities) if similarity >= threshold]

    edge_indices = edge_indices.detach().cpu().tolist()
    from_indices, to_indices = edge_indices[0], edge_indices[1]
    edges = edge_indices_to_edges(from_indices, to_indices)

    positive_edges = [edge for idx, edge in enumerate(edges) if idx in positive_edge_indices]
    return positive_edges


def join_bites(
        bites: List[Bite],
        joiner: JoinerGraphModel,
        graph_provider: JoinerGraphProvider,
        normalizer: GraphNormalizer,
        filename: str,
        pagexml: PageLayout,
        device,
        threshold: float
        ) -> List[Bite]:
    graph = graph_provider.get_graph_from_bites(bites, filename, pagexml)

    normalizer.normalize_graphs([graph])
    positive_edges = get_positive_edges_gcn(graph, joiner, device, threshold)

    nodes_to_join = get_joining_dict(positive_edges)
    new_bites = join_bites_by_dict(nodes_to_join, bites)

    return new_bites
