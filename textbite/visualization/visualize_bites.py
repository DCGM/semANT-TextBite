import logging
import sys
import argparse
import os
from typing import List

import cv2
from cv2.typing import MatLike
import numpy as np

from pero_ocr.core.layout import PageLayout

from textbite.bite import load_bites, Bite


ALPHA = 0.4



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


def parse_arguments():
    print(' '.join(sys.argv), file=sys.stderr)

    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--images", required=True, type=str, help="Path to a folder with jpg data.")
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xml data.")
    parser.add_argument("--jsons", required=True, type=str, help="Path to a folder with json data.")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put the outputs")

    args = parser.parse_args()
    return args


def overlay_line(img, line, color, alpha):
    mask = np.zeros_like(img)
    pts = line.polygon
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], color)
    return cv2.addWeighted(img, 1, mask, 1-alpha, 0)


def draw_bites(img: MatLike, bites: List[Bite]) -> MatLike:
    for idx, bite in enumerate(bites):
        start_x = int(bite.bbox.xmin)
        end_x = int(bite.bbox.xmax)
        start_y = int(bite.bbox.ymin)
        end_y = int(bite.bbox.ymax)
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), COLORS[idx % len(COLORS)], 10)

    return img


def overlay_bites_lines(img: MatLike, pagexml: PageLayout, bites: List[Bite]) -> MatLike:
    overlay = np.zeros_like(img)

    for line in pagexml.lines_iterator():
        if not line.transcription.strip():
            continue
        
        line_found = False
        for bite_idx, bite in enumerate(bites):
            if line.id in bite.lines:
                line_found = True
                overlay = overlay_line(overlay, line, COLORS[bite_idx % len(COLORS)], ALPHA)

        if not line_found:
            logging.warning(f"Line {line.id} with transcription {repr(line.transcription)} not found in any bite.")
            continue

    return cv2.addWeighted(img, 1, overlay, 1-ALPHA, 0)


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)

    os.makedirs(args.save, exist_ok=True)

    json_filenames = [json_filename for json_filename in os.listdir(args.jsons) if json_filename.endswith(".json")]

    for json_filename in json_filenames:
        path_json = os.path.join(args.jsons, json_filename)
        path_xml = os.path.join(args.xml, json_filename.replace(".json", ".xml"))
        path_img = os.path.join(args.images, json_filename.replace(".json", ".jpg"))

        try:
            pagexml = PageLayout(file=path_xml)
        except OSError:
            logging.warning(f"XML {path_xml} not found, skipping.")
            continue

        try:
            img = cv2.imread(path_img)
        except OSError:
            logging.warning(f"Image {path_img} not found, skipping.")
            continue

        bites = load_bites(path_json)

        # img_bites = draw_bites(img, bites)
        result = overlay_bites_lines(img, pagexml, bites)

        res_filename = os.path.join(args.save, json_filename.replace(".json", "-bites.jpg"))
        cv2.imwrite(res_filename, result)

        logging.info(f'Processed {json_filename}')


if __name__ == "__main__":
    main()
