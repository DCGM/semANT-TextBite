import logging
import sys
import argparse
import os
import lxml.etree as ET
from itertools import pairwise

import cv2
from cv2.typing import MatLike
import numpy as np

from pero_ocr.document_ocr.layout import points_string_to_array

from textbite.geometry import polygon_centroid


COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (34, 139, 34),    # Forest Green
    (70, 130, 180),   # Steel Blue
    (255, 20, 147),   # Deep Pink
    (218, 112, 214),  # Orchid
    (255, 165, 0),    # Orange
    (173, 216, 230),  # Light Blue
    (255, 69, 0),     # Red-Orange
    (0, 191, 255),    # Deep Sky Blue
    (128, 0, 128),    # Purple
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 99, 71),    # Tomato
    (255, 192, 203),  # Pink
    (32, 178, 170),   # Light Sea Green
    (250, 128, 114),  # Salmon
    (0, 128, 128),    # Teal
    (240, 230, 140)   # Khaki
]


ALPHA = 0.3


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--xml-input", required=True, type=str, help="Path to a folder with xml data of transcribed pages.")
    parser.add_argument("--images", required=True, type=str, help="Path to a folder with images data.")
    parser.add_argument("--images-output", type=str, required=True, help="Where to put visualized xmls.")

    args = parser.parse_args()
    return args


def array_from_elem(elem, namespace):
    polygon_str = elem.find(".//ns:Coords", namespace).get("points")
    return points_string_to_array(polygon_str)


def draw_polygon(img, polygon, color, alpha):
    mask = np.zeros_like(img)
    polygon = polygon.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [polygon], color)
    return cv2.addWeighted(img, 1, mask, 1-alpha, 0)


def draw_reading_order(img: MatLike, reading_order, centers: dict) -> MatLike:
    color = (255, 0, 0)
    for ordered_group in reading_order:
        for src, tgt in pairwise(ordered_group):
            from_point = centers[src]
            to_point = centers[tgt]
            cv2.arrowedLine(img, from_point, to_point, color=color, thickness=10, tipLength=0.05)

    return img


def draw_layout(img: MatLike, root) -> MatLike:
    overlay = np.zeros_like(img)

    ns_name = root.nsmap[None]
    namespace = {"ns": ns_name}

    region_centers = {}

    for region_idx, region in enumerate(root.iter(f"{{{ns_name}}}TextRegion")):
        color = COLORS[region_idx % len(COLORS)]

        region_polygon = array_from_elem(region, namespace)
        region_centers[region.get("id")] = [int(item) for item in polygon_centroid(region_polygon[:, 0], region_polygon[:, 1])]

        cv2.drawContours(img, [region_polygon], -1, color=color, thickness=10)

        for line in region.iter(f"{{{ns_name}}}TextLine"):
            line_polygon = array_from_elem(line, namespace)
            overlay = draw_polygon(overlay, line_polygon, color=color, alpha=ALPHA)

    reading_order_elem = root.find(".//ns:ReadingOrder", namespace)
    if len(reading_order_elem) > 1:
        logging.warning("Reading order has multiple groups, taking the first one.")

    reading_order = []
    for group in reading_order_elem.iter(f"{{{ns_name}}}OrderedGroup"):
        if len(group) < 2:
            continue

        group_reading_order = [elem.get("regionRef") for elem in group.iter(f"{{{ns_name}}}RegionRefIndexed")]
        reading_order.append(group_reading_order)
    
    draw_reading_order(img, reading_order, region_centers)

    return cv2.addWeighted(img, 1, overlay, 1-ALPHA, 0)


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)

    os.makedirs(args.images_output, exist_ok=True)

    xml_filenames = [xml_filename for xml_filename in os.listdir(args.xml_input) if xml_filename.endswith(".xml")]

    for xml_filename in xml_filenames:
        logging.info(f"Processing {xml_filename} ...")
        path_xml = os.path.join(args.xml_input, xml_filename)
        with open(path_xml, "r") as f:
            root = ET.fromstring(f.read())

        image_filename = xml_filename.replace(".xml", ".jpg")
        path_img = os.path.join(args.images, image_filename)
        img = cv2.imread(path_img)
        if img is None:
            logging.warning(f"Image {image_filename} not found, skipping.")

        result = draw_layout(img, root)

        res_path = os.path.join(args.images_output, image_filename)
        cv2.imwrite(res_path, result)


if __name__ == "__main__":
    main()
