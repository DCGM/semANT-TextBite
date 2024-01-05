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

    parser.add_argument("--overlay", action='store_true', help="Whether to overlay regions by colors")
    parser.add_argument("--draw-out-of-order", action='store_true', help="Whether to draw regions that are not part of reading order")
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


def load_reading_order(root, namespace, ns_name):
    reading_order_elem = root.find(".//ns:ReadingOrder", namespace)
    if reading_order_elem is None:
        return []

    if len(reading_order_elem) > 1:
        logging.warning("Reading order has multiple groups, taking the first one.")

    reading_order = []
    for group in reading_order_elem.iter(f"{{{ns_name}}}OrderedGroup"):
        group_reading_order = [elem.get("regionRef") for elem in group.iter(f"{{{ns_name}}}RegionRefIndexed")]
        reading_order.append(group_reading_order)

    return reading_order


def load_regions(root, namespace, ns_name):
    region_centers = {}
    region_polygons = {}
    region_lines_polygons = {}
    for region in root.iter(f"{{{ns_name}}}TextRegion"):
        polygon = array_from_elem(region, namespace)
        region_centers[region.get("id")] = [int(item) for item in polygon_centroid(polygon[:, 0], polygon[:, 1])]
        region_polygons[region.get("id")] = polygon
        region_lines_polygons[region.get("id")] = [array_from_elem(line, namespace) for line in region.iter(f"{{{ns_name}}}TextLine")]

    return region_polygons, region_centers, region_lines_polygons


class ImageOverdrawer:
    def __init__(self, img, draw_overlay):
        self.img = img
        self.draw_overlay = draw_overlay
        self.overlay = np.zeros_like(img) if draw_overlay else None

    def draw_region(self, color, polygon, lines_polygons):
        cv2.drawContours(self.img, [polygon], -1, color=color, thickness=10)

        if self.draw_overlay:
            for line_polygon in lines_polygons:
                self.overlay = draw_polygon(self.overlay, line_polygon, color=color, alpha=ALPHA)

    def connect_regions(self, color, from_point, to_point):
        cv2.arrowedLine(self.img, from_point, to_point, color=color, thickness=10, tipLength=0.05)

    def final_img(self):
        if self.draw_overlay:
            return cv2.addWeighted(self.img, 1, self.overlay, 1-ALPHA, 0)
        else:
            return self.img


def draw_layout(drawer, root, draw_out_of_order) -> MatLike:
    ns_name = root.nsmap[None]
    namespace = {"ns": ns_name}

    region_polygons, region_centers, region_lines_polygons = load_regions(root, namespace, ns_name)
    reading_order = load_reading_order(root, namespace, ns_name)

    for bite_id, bite in enumerate(reading_order):
        color = COLORS[bite_id % len(COLORS)]
        for region_id in bite:
            polygon, lines_polygons = region_polygons[region_id], region_lines_polygons[region_id]
            drawer.draw_region(color, polygon, lines_polygons)

        for src, tgt in pairwise(bite):
            drawer.connect_regions(color, region_centers[src], region_centers[tgt])

    if draw_out_of_order:
        regions_out_of_order = set(region_polygons.keys()) - set(r for bite in reading_order for r in bite)
        for i, region_id in enumerate(regions_out_of_order):
            color = COLORS[(len(reading_order) + i) % len(COLORS)]
            polygon, lines_polygons = region_polygons[region_id], region_lines_polygons[region_id]
            drawer.draw_region(color, polygon, lines_polygons)

    return drawer.final_img()


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
            continue

        drawer = ImageOverdrawer(img, args.overlay)
        result = draw_layout(drawer, root, args.draw_out_of_order)

        res_path = os.path.join(args.images_output, image_filename)
        cv2.imwrite(res_path, result)


if __name__ == "__main__":
    main()
