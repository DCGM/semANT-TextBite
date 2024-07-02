"""Utilities for feature extraction

Date -- 15.05.2024
Author -- Martin Kostelnik
"""


from typing import List, Optional

import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizerFast, BertModel

from pero_ocr.document_ocr.layout import PageLayout

from textbite.geometry import dist_l2, PageGeometry, RegionGeometry


class TextFeaturesProvider:
    def __init__(
        self,
        tokenizer: Optional[BertTokenizerFast]=None,
        czert: Optional[BertModel]=None,
        device=None,
        ):
        self.tokenizer = tokenizer
        self.czert = czert
        self.device = device

    def get_tfidf_features(self, transcriptions: List[str], max_features: int=64) -> List[torch.FloatTensor]:
        vectorizer = TfidfVectorizer(max_features=max_features)

        try:
            tfidf_matrix = vectorizer.fit_transform(transcriptions)
        except ValueError:
            raise RuntimeError("XML Transcriptions corrupted")
        
        tfidf_list = tfidf_matrix.todense(order="C").tolist()

        for item in tfidf_list:
            if len(item) != max_features:
                to_add = max_features - len(item)
                item.extend([0.0] * to_add)

        tfidf_list = [torch.tensor(features, dtype=torch.float32) for features in tfidf_list]

        return tfidf_list
    
    def get_czert_features(self, text: str):
        assert self.czert is not None
        assert self.tokenizer is not None

        text = text.replace("\n", " ").strip()

        tokenized_text = self.tokenizer(
            text,
            max_length=512,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = tokenized_text["input_ids"].to(self.device)
        token_type_ids = tokenized_text["token_type_ids"].to(self.device)
        attention_mask = tokenized_text["attention_mask"].to(self.device)

        with torch.no_grad():
            czert_outputs = self.czert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pooler_output = czert_outputs.pooler_output
        cls_output = czert_outputs.last_hidden_state[:, 0, :]

        return pooler_output, cls_output


class GeometryFeaturesProvider:
    def __init__(self):
        self.pagexml = None
        self.page_geometry = None

    def get_regions_features(self, geometry: PageGeometry, pagexml: PageLayout) -> List[torch.FloatTensor]:
        self.pagexml = pagexml
        return [self.get_region_features(region) for region in geometry.regions]

    def get_region_features(self, region_geometry: RegionGeometry) -> torch.FloatTensor:
        assert self.pagexml is not None
        page_height, page_width = self.pagexml.page_size

        region = region_geometry.bbox

        feature_center_x = region_geometry.center.x
        feature_center_y = region_geometry.center.y
        feature_center_x_relative = region_geometry.center.x / page_width
        feature_center_y_relative = region_geometry.center.y / page_height

        feature_xmin = float(region.xmin)
        feature_xmax = float(region.xmax)
        feature_ymin = float(region.ymin)
        feature_ymax = float(region.ymax)

        feature_xmin_relative = feature_xmin / page_width
        feature_xmax_relative = feature_xmax / page_width
        feature_ymin_relative = feature_ymin / page_height
        feature_ymax_relative = feature_ymax / page_height

        feature_area = region_geometry.bbox_area
        feature_area_relative = feature_area / (page_width * page_height)

        feature_width = region_geometry.width
        feature_width_relative = feature_width / page_width

        feature_height = region_geometry.height
        feature_height_relative = feature_height / page_height

        feature_ratio = feature_width / feature_height

        feature_number_of_predecessors = float(region_geometry.number_of_predecessors)
        feature_number_of_successors = float(region_geometry.number_of_successors)

        feature_area_relative_to_parent = 0.0
        feature_area_relative_to_child = 0.0

        feature_width_relative_to_parent = 0.0
        feature_width_relative_to_child = 0.0
        
        feature_distance_to_parent_y = 0.0
        feature_distance_to_child_y = 0.0
        
        feature_distance_to_parent_y_relative = 0.0
        feature_distance_to_child_y_relative = 0.0

        if region_geometry.parent is not None:
            parent = region_geometry.parent
            feature_area_relative_to_parent = feature_area / parent.bbox_area
            feature_width_relative_to_parent = region_geometry.width / parent.width
            feature_distance_to_parent_y = max(0.0, region.ymin - parent.bbox.ymax)
            feature_distance_to_parent_y_relative = feature_distance_to_parent_y / page_height

        if region_geometry.child is not None:
            child = region_geometry.child
            feature_area_relative_to_child = feature_area / child.bbox_area
            feature_width_relative_to_child = region_geometry.width / child.width   
            feature_distance_to_child_y = max(0.0, child.bbox.ymin - region.ymax)
            feature_distance_to_child_y_relative = feature_distance_to_child_y / page_height

        features = [
            feature_center_x,
            feature_center_y,
            feature_xmin,
            feature_xmax,
            feature_ymin,
            feature_ymax,
            feature_area,
            feature_width,
            feature_height,
            feature_ratio,
            feature_area_relative,
            feature_width_relative,
            feature_height_relative,
            feature_center_x_relative,
            feature_center_y_relative,
            feature_xmin_relative,
            feature_xmax_relative,
            feature_ymin_relative,
            feature_ymax_relative,
            feature_number_of_predecessors,
            feature_number_of_successors,
            feature_area_relative_to_parent,
            feature_area_relative_to_child,
            feature_width_relative_to_parent,
            feature_width_relative_to_child,
            feature_distance_to_parent_y,
            feature_distance_to_child_y,
            feature_distance_to_child_y_relative,
            feature_distance_to_parent_y_relative,
        ]

        return torch.tensor(features, dtype=torch.float32)

    def get_features(self, return_type: Optional[str]=None) -> List | np.ndarray | torch.FloatTensor:
        assert self.page_geometry, "Cannot determine features without data of the entire page"
        PLACEHOLDER_VALUE = -100.0

        features = []

        width = self.get_width()
        height = self.get_height()

        #  1. Line width relative to it's neighbourhood average
        vertical_neighbourhood = self.get_vertical_neighbours(2)
        vertical_neighbourhood_avg_width = sum([l.get_width() for l in vertical_neighbourhood]) / len(vertical_neighbourhood)
        features.append(width / vertical_neighbourhood_avg_width)

        #  2. Line width relative to page average
        #  3. Line height relative to page average
        features.append(width / self.page_geometry.get_avg_line_width())
        features.append(height / self.page_geometry.get_avg_line_height())

        # 10. Line width relative to page width
        # 11. Line height relative to page height
        features.append(width / self.page_geometry.page_width)
        features.append(height / self.page_geometry.page_height)

        # 12. Center X coordinate relative to page width
        # 13. Center Y coordinate relative to page height
        features.append(self.center.x / self.page_geometry.page_width)
        features.append(self.center.y / self.page_geometry.page_height)

        # 22. Left X coordinate relative to page width
        # 23. Right X coordinate relative to page width
        # 24. Top Y coordinate relative to page height
        # 25. Bottom Y coordinate relative to page height
        features.append(self.bbox.xmin / self.page_geometry.page_width)
        features.append(self.bbox.xmax / self.page_geometry.page_width)
        features.append(self.bbox.ymin / self.page_geometry.page_height)
        features.append(self.bbox.ymax / self.page_geometry.page_height)

        # 21. Line width relative to line height
        features.append(width / height)

        # 18. Bounding box area relative to page area
        features.append(self.bbox_area / self.page_geometry.page_area)

        #  4. L2 distance from parent (center - center)
        #  6. L2 distance from parent (center - center) relative to my height
        # 14. Distance from parent in Y axis (center - center)
        # 16. Distance from parent in X axis (center - center)
        # 19. Bounding box area relative to parent bounding box area
        if self.parent:
            distance_to_parent = dist_l2(self.center, self.parent.center)
            features.append(distance_to_parent)
            features.append(distance_to_parent / height)
            features.append(abs(self.center.y - self.parent.center.y))
            features.append(abs(self.center.x - self.parent.center.x))
            features.append(self.bbox_area / self.parent.bbox_area)
        else:
            features.extend([PLACEHOLDER_VALUE] * 5)

        #  5. L2 distance from child (center - center)
        #  7. L2 distance from child (center - center) relative to my height
        # 15. Distance from child in Y axis (center - center)
        # 17. Distance from child in X axis (center - center)
        # 20. Bounding box area relative to child bounding box area
        if self.child:
            distance_to_child = dist_l2(self.center, self.child.center)
            features.append(distance_to_child)
            features.append(distance_to_child / height)
            features.append(abs(self.center.y - self.child.center.y))
            features.append(abs(self.center.x - self.child.center.x))
            features.append(self.bbox_area / self.child.bbox_area)
        else:
            features.extend([PLACEHOLDER_VALUE] * 5)

        #  8. Number of predecessors
        features.append(self.get_number_of_predecessors())

        #  9. Number of children
        features.append(self.get_number_of_successors())

        match return_type:
            case "pt":
                return torch.tensor(features, dtype=torch.float32)

            case "np":
                return np.array(features, dtype=np.float32)

            case _:
                return features
