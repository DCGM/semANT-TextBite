from typing import List, Optional

import torch
from transformers import BertTokenizerFast, BertModel

from pero_ocr.core.layout import PageLayout

from textbite.geometry import PageGeometry, RegionGeometry


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
