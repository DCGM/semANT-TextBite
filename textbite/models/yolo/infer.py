"""Inference using the YOLO model

Date -- 15.05.2024
Author -- Martin Kostelnik
"""


import argparse
import json
import logging
import os.path
from typing import List, Tuple, Optional

from numba.core.errors import NumbaDeprecationWarning
import warnings
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)

import xml.etree.ElementTree as ET
from pero_ocr.document_ocr.layout import PageLayout
from ultralytics import YOLO
from safe_gpu import safe_gpu

from textbite.bite import Bite
from textbite.geometry import AABB, polygon_to_bbox, bbox_intersection_over_area, best_intersecting_bbox
from textbite.models.improve_pagexml import PageXMLEnhancer, UnsupportedLayoutError


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--data", required=True, type=str, help="Path to a folder with xml data.")
    parser.add_argument("--images", required=True, type=str, help="Path to a folder with images data.")
    parser.add_argument("--altos", type=str, help="Path to a folder with alto data.")
    parser.add_argument("--xml-output", type=str, help="Where to put reorganized PAGE XMLs.")
    parser.add_argument("--model", required=True, type=str, help="Path to the .pt file with weights of YOLO model.")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put output jsons.")

    return parser.parse_args()


class YoloBiter:
    def __init__(self, model):
        self.model = model

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

    def is_contained(self, lhs: AABB, rhs: AABB, threshold: float=0.9) -> bool:
        return bbox_intersection_over_area(lhs, rhs) >= threshold

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
                if i != j and self.is_contained(box1, box2):
                    is_enclosed = True
                    break

            if not is_enclosed:
                new_bboxes.append(box1)

        return new_bboxes

    def produce_bites(self, img_filename: str, layout: PageLayout, alto_filename: Optional[str]=None) -> List[Bite]:
        texts, titles = self.find_bboxes(img_filename)

        texts = self.filter_bboxes(texts)
        titles = self.filter_bboxes(titles)

        if alto_filename:
            alto_tree = ET.parse(alto_filename)
            alto_root = alto_tree.getroot()
            namespace = {"ns": "http://www.loc.gov/standards/alto/ns-v2#"}
            alto_text_lines = alto_root.findall(".//ns:TextLine", namespace)
            alto_text_lines_bboxes = [self.get_alto_bbox(atl) for atl in alto_text_lines]

        texts_dict = {idx: Bite(cls="text", bbox=bbox) for idx, bbox in enumerate(texts)}
        titles_dict = {idx: Bite(cls="title", bbox=bbox) for idx, bbox in enumerate(titles)}
        for line in layout.lines_iterator():
            line_bbox = polygon_to_bbox(line.polygon)
            best_text_idx = best_intersecting_bbox(line_bbox, texts)
            best_title_idx = best_intersecting_bbox(line_bbox, titles)
            if best_text_idx is None and best_title_idx is None:
                continue

            if alto_filename:
                best_alto_idx = best_intersecting_bbox(line_bbox, alto_text_lines_bboxes)
                alto_possible = best_alto_idx is not None
                if alto_possible:
                    alto_text_line = alto_text_lines[best_alto_idx]
                    alto_words = alto_text_line.findall(".//ns:String", namespace)

            best_text_ioa = 0.0 if best_text_idx is None else bbox_intersection_over_area(line_bbox, texts[best_text_idx])
            best_title_ioa = 0.0 if best_title_idx is None else bbox_intersection_over_area(line_bbox, titles[best_title_idx])

            if best_title_idx is not None and (best_text_ioa < 0.2 or best_text_idx is None):
                titles_dict[best_title_idx].lines.append(line.id)
                continue

            if best_text_idx is not None and (best_title_ioa < 0.2 or best_title_idx is None):
                texts_dict[best_text_idx].lines.append(line.id)
                continue

            texts_dict[best_text_idx].lines.append(line.id)
            if alto_filename and alto_possible:
                for word in alto_words:
                    xmin = float(word.get("HPOS"))
                    ymin = float(word.get("VPOS"))
                    xmax = xmin + float(word.get("WIDTH"))
                    ymax = ymin + float(word.get("HEIGHT"))
                    word_bbox = AABB(xmin, ymin, xmax, ymax)
                    word_ioa = bbox_intersection_over_area(word_bbox, titles[best_title_idx])
                    if word_ioa > 0.2:
                        texts_dict[best_text_idx].name += f"{word.get('CONTENT')} "

                texts_dict[best_text_idx].name = texts_dict[best_text_idx].name.strip()
                texts_dict[best_text_idx].name = texts_dict[best_text_idx].name.strip(r".:;,?!'\"!@#$%/+-^&*)(")

        texts = [bite for bite in texts_dict.values() if bite.lines]
        titles = [bite for bite in titles_dict.values() if bite.lines]

        return texts + titles


def save_result(result: List[Bite], path: str) -> None:
    with open(path, "w") as f:
        json.dump([bite.__dict__ for bite in result], f, indent=4, ensure_ascii=False)


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    safe_gpu.claim_gpus()

    biter = YoloBiter(YOLO(args.model))
    xml_enhancer = PageXMLEnhancer()

    os.makedirs(args.save, exist_ok=True)
    xml_filenames = [xml_filename for xml_filename in os.listdir(args.data) if xml_filename.endswith(".xml")]

    for filename in xml_filenames:
        path_xml = os.path.join(args.data, filename)
        layout = PageLayout()
        with open(path_xml) as f:
            layout.from_pagexml(f)
        xml_enhancer.ensure_unique_line_ids(layout)

        path_img = os.path.join(args.images, filename.replace(".xml", ".jpg"))
        path_alto = os.path.join(args.altos, filename) if args.altos else None
        logging.info(f"Processing: {path_xml}")
        bites = biter.produce_bites(path_img, layout, path_alto)

        out_path = os.path.join(args.save, filename.replace(".xml", ".json"))
        save_result(bites, out_path)

        try:
            out_xml_string = xml_enhancer.process(layout, bites)
            out_path = os.path.join(args.save, filename)
            with open(out_path, 'w', encoding='utf-8') as out_f:
                out_f.write(out_xml_string)
        except UnsupportedLayoutError as e:
            logging.warning(e)
        except ValueError:
            continue


if __name__ == '__main__':
    main()
