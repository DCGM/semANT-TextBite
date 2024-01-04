#!/usr/bin/env python3

import argparse
import json
import logging
import os.path
from typing import List
import requests
import sys

from pero_ocr.document_ocr.layout import PageLayout

from textbite.improve_pagexml import PageXMLEnhancer, UnsupportedLayoutError
from textbite.detection import YoloBiter, Bite


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--xml-input", required=True, type=str, help="Path to a folder with xml data of transcribed pages.")
    parser.add_argument("--images", required=True, type=str, help="Path to a folder with images data.")
    parser.add_argument("--altos", type=str, help="Path to a folder with alto data.")
    parser.add_argument("--model", type=str, help="Path to the .pt file with weights of YOLO model.")
    parser.add_argument("--xml-output", type=str, required=True, help="Where to put reorganized PAGE XMLs.")
    parser.add_argument("--bites-out", type=str, help="Folder where to put output TextBites as raw jsons.")

    return parser.parse_args()


def save_result(result: List[Bite], path: str) -> None:
    with open(path, "w") as f:
        json.dump([bite.__dict__ for bite in result], f, indent=4, ensure_ascii=False)


default_model_url = 'https://nextcloud.fit.vutbr.cz/s/5xHQcgosNai9pwa/download/textbite-detector.pt'
cache_path = os.path.join(os.path.expanduser('~'), '.cache', 'textbite')
cached_model_path = os.path.join(cache_path, 'default_model.pt')


def get_default_model(force_download=False):
    if os.path.isfile(cached_model_path) and not force_download:
        logging.info('Default model already present')
        return

    logging.info('Downloading default model...')
    r = requests.get(default_model_url)
    os.makedirs(cache_path, exist_ok=True)
    with open(cached_model_path, 'wb') as f:
        f.write(r.content)


def create_biter(model_path):
    if model_path:
        logging.info(f"Loading model from {model_path}...")
        return YoloBiter(model_path)
    else:
        get_default_model()
        try:
            return YoloBiter(cached_model_path)
        except Exception:
            logging.warning(f"Failed to load the model from '{cached_model_path}'. Retrying with forced download")

        get_default_model(force_download=True)
        return YoloBiter(cached_model_path)


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level, force=True)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    biter = create_biter(args.model)
    xml_enhancer = PageXMLEnhancer()

    os.makedirs(args.xml_output, exist_ok=True)
    if args.bites_out:
        os.makedirs(args.bites_out, exist_ok=True)

    xml_filenames = [xml_filename for xml_filename in os.listdir(args.xml_input) if xml_filename.endswith(".xml")]

    for filename in xml_filenames:
        path_xml = os.path.join(args.xml_input, filename)

        layout = PageLayout()
        with open(path_xml) as f:
            layout.from_pagexml(f)
        xml_enhancer.ensure_unique_line_ids(layout)

        path_img = os.path.join(args.images, filename.replace(".xml", ".jpg"))
        path_alto = os.path.join(args.altos, filename) if args.altos else None

        logging.info(f"Processing: {path_xml}")
        bites = biter.produce_bites(path_img, layout, path_alto)

        if args.bites_out:
            out_path = os.path.join(args.bites_out, filename.replace(".xml", ".json"))
            save_result(bites, out_path)

        try:
            out_xml_string = xml_enhancer.process(layout, bites)
            out_path = os.path.join(args.xml_output, filename)
            with open(out_path, 'w', encoding='utf-8') as out_f:
                out_f.write(out_xml_string)
        except UnsupportedLayoutError as e:
            logging.warning(e)
            continue


if __name__ == '__main__':
    sys.exit(main())
