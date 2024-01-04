from typing import List, Tuple, Optional, Any
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from dataclasses import dataclass, field

from pero_ocr.document_ocr.layout import PageLayout

from textbite.geometry import AABB, polygon_to_bbox, bbox_intersection_over_area, best_intersecting_bbox, \
                              is_contained, PageGeometry


@dataclass
class Bite:
    cls: str
    bbox: Optional[AABB]
    lines: List[str] = field(default_factory=list)
    name: str = ""


class YoloBiter:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def find_bboxes(self, img_filename: str) -> Tuple[List[AABB], List[AABB]]:
        results = self.model.predict(source=img_filename)
        assert len(results) == 1
        result = results[0]

        cls_indices = {v: k for k, v in result.names.items()}
        title_index = cls_indices["title"]
        text_index = cls_indices["text"]

        texts_ = [b for b in result.boxes if b.cls == text_index]
        titles_ = [b for b in result.boxes if b.cls == title_index]

        texts = []
        titles = []

        for text in texts_:
            texts.append(AABB(*text.xyxy[0].cpu().numpy().tolist()))

        for title in titles_:
            titles.append(AABB(*title.xyxy[0].cpu().numpy().tolist()))

        return texts, titles

    def get_alto_bbox(self, alto_line) -> AABB:
        xmin = float(alto_line.get("HPOS"))
        ymin = float(alto_line.get("VPOS"))
        xmax = xmin + float(alto_line.get("WIDTH"))
        ymax = ymin + float(alto_line.get("HEIGHT"))

        return AABB(xmin, ymin, xmax, ymax)

    def filter_bboxes(self, bboxes: List[AABB]) -> List[AABB]:
        new_bboxes = []

        for i, box1 in enumerate(bboxes):
            is_enclosed = False
            for j, box2 in enumerate(bboxes):
                if i != j and is_contained(box1, box2):
                    is_enclosed = True
                    break

            if not is_enclosed:
                new_bboxes.append(box1)

        return new_bboxes

    def produce_bites(self, img_filename: str, layout: PageLayout, alto_filename: Optional[str]=None) -> List[Bite]:
        texts_bboxes, titles_bboxes = self.find_bboxes(img_filename)

        texts_bboxes = self.filter_bboxes(texts_bboxes)
        titles_bboxes = self.filter_bboxes(titles_bboxes)

        if alto_filename:
            alto_tree = ET.parse(alto_filename)
            alto_root = alto_tree.getroot()
            namespace = {"ns": "http://www.loc.gov/standards/alto/ns-v2#"}
            alto_text_lines = alto_root.findall(".//ns:TextLine", namespace)
            alto_text_lines_bboxes = [self.get_alto_bbox(atl) for atl in alto_text_lines]

        texts_dict = {idx: Bite(cls="text", bbox=bbox) for idx, bbox in enumerate(texts_bboxes)}
        titles_dict = {idx: Bite(cls="title", bbox=bbox) for idx, bbox in enumerate(titles_bboxes)}
        for line in layout.lines_iterator():
            line_bbox = polygon_to_bbox(line.polygon)
            best_text_idx = best_intersecting_bbox(line_bbox, texts_bboxes)
            best_title_idx = best_intersecting_bbox(line_bbox, titles_bboxes)
            if best_text_idx is None and best_title_idx is None:
                continue

            if alto_filename:
                best_alto_idx = best_intersecting_bbox(line_bbox, alto_text_lines_bboxes)
                alto_possible = best_alto_idx is not None
                if alto_possible:
                    alto_text_line = alto_text_lines[best_alto_idx]
                    alto_words = alto_text_line.findall(".//ns:String", namespace)

            best_text_ioa = 0.0 if best_text_idx is None else bbox_intersection_over_area(line_bbox, texts_bboxes[best_text_idx])
            best_title_ioa = 0.0 if best_title_idx is None else bbox_intersection_over_area(line_bbox, titles_bboxes[best_title_idx])

            if best_title_idx is not None and (best_text_ioa < 0.2 or best_text_idx is None):
                titles_dict[best_title_idx].lines.append(line.id)
                continue

            if best_text_idx is not None and (best_title_ioa < 0.2 or best_title_idx is None):
                texts_dict[best_text_idx].lines.append(line.id)
                continue

            texts_dict[best_text_idx].lines.append(line.id)
            if alto_filename and alto_possible:
                for word in alto_words:
                    word_bbox = self.get_alto_bbox(word)
                    word_ioa = bbox_intersection_over_area(word_bbox, titles_bboxes[best_title_idx])
                    if word_ioa > 0.2:
                        texts_dict[best_text_idx].name += f"{word.get('CONTENT')} "

                texts_dict[best_text_idx].name = texts_dict[best_text_idx].name.strip()
                texts_dict[best_text_idx].name = texts_dict[best_text_idx].name.strip(r".:;,?!'\"!@#$%/+-^&*)(")

        texts = [bite for bite in texts_dict.values() if bite.lines]
        titles = [bite for bite in titles_dict.values() if bite.lines]

        # Join titles with their children bites
        remaining_titles = []
        geometry = PageGeometry(texts + titles)

        for bg in geometry.bite_geometries:
            if bg.bite.cls != "title":
                continue
        
            if bg.child:
                y_dist = abs(bg.bite.bbox.ymax - bg.child.bite.bbox.ymin)

            found_text_successor = False
            for successor in bg.children_iterator():
                if successor.bite.cls == "text" and y_dist < 0.1 * layout.page_size[0]:
                    found_text_successor = True
                    successor.bite.lines.extend(bg.bite.lines)
                    break

            if not found_text_successor:
                remaining_titles.append(bg.bite)
                continue

        return texts + remaining_titles
