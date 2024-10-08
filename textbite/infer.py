import argparse
import os
import logging
import pickle
from collections import namedtuple
import requests

from safe_gpu import safe_gpu
from pero_ocr.core.layout import PageLayout
from ultralytics import YOLO
import torch
from transformers import BertTokenizerFast, BertModel

from textbite.models.yolo.infer import YoloBiter
from textbite.models.utils import GraphNormalizer
from textbite.models.joiner.graph import JoinerGraphProvider
from textbite.models.joiner.model import JoinerGraphModel
from textbite.models.joiner.infer import join_bites
from textbite.models.improve_pagexml import PageXMLEnhancer, UnsupportedLayoutError
from textbite.bite import save_bites


IMAGE_EXTENSIONS = [".jpg", ".jpeg"]
CLASSIFICATION_THRESHOLD = 0.68
CZERT_PATH = r"UWB-AIR/Czert-B-base-cased"

ModelPart = namedtuple("ModelPart", "name url cached_path")
CACHE_PATH = os.path.join(os.path.expanduser('~'), '.cache', 'textbite')
YOLO_URL = r"https://nextcloud.fit.vutbr.cz/s/KwmAa6gHopze2iz/download/yolo-s-800.pt"
GNN_URL = r"https://nextcloud.fit.vutbr.cz/s/gL9zyaHJjTMfS7B/download/gnn-joiner.pth"
NORMALIZER_URL = r"https://nextcloud.fit.vutbr.cz/s/LsZ5nG5HGwjcBwW/download/gnn-normalizer.pkl"
YOLO_CACHED_PATH = os.path.join(CACHE_PATH, "yolo-s-800.pt")
GNN_CACHED_PATH = os.path.join(CACHE_PATH, "gnn-joiner.pth")
NORMALIZER_CACHED_PATH = os.path.join(CACHE_PATH, "gnn-normalizer.pkl")
YOLO_ = ModelPart("yolo", YOLO_URL, YOLO_CACHED_PATH)
GNN = ModelPart("gnn", GNN_URL, GNN_CACHED_PATH)
NORMALIZER = ModelPart("normalizer", NORMALIZER_URL, NORMALIZER_CACHED_PATH)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logging-level", default='WARNING', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'])
    parser.add_argument("--xml", required=True, type=str, help="Path to a folder with xml data.")
    parser.add_argument("--img", required=True, type=str, help="Path to a folder with images data.")
    parser.add_argument("--alto", default=None, type=str, help="Path to a folder with alto data.")
    parser.add_argument("--yolo", default=None, type=str, help="Path to the .pt file with weights of YOLO model.")
    parser.add_argument("--gnn", default=None, type=str, help="Path to the .pt file with weights of Joiner model.")
    parser.add_argument("--normalizer", default=None, type=str, help="Path to node normalizer.")
    parser.add_argument("--czert", default=CZERT_PATH, type=str, help="Path to CZERT.")
    parser.add_argument("--json", action="store_true", help="Store the JSON output format")
    parser.add_argument("--save", required=True, type=str, help="Folder where to put output files.")

    return parser.parse_args()


def download_model_part(model_part: ModelPart, force_download: bool=False):
    if os.path.isfile(model_part.cached_path) and not force_download:
        logging.info(f"Default {model_part.name} model already present")
        return

    logging.info(f'Downloading default {model_part.name} model...')
    r = requests.get(model_part.url)
    os.makedirs(CACHE_PATH, exist_ok=True)
    with open(model_part.cached_path, 'wb') as f:
        f.write(r.content)


def create_models(args):
    if args.yolo is None:
        download_model_part(YOLO_)
        yolo = YoloBiter(YOLO(YOLO_.cached_path))
    else:
        yolo = YoloBiter(YOLO(args.yolo))

    if args.gnn is None:
        download_model_part(GNN)
        gnn = torch.load(GNN.cached_path)
    else:
        gnn = torch.load(args.gnn)

    if args.normalizer is None:
        download_model_part(NORMALIZER)
        normalizer_path = NORMALIZER.cached_path
    else:
        normalizer_path = args.normalizer

    with open(normalizer_path, "rb") as f:
        normalizer = pickle.load(f)

    return yolo, gnn, normalizer


def main():
    args = parse_arguments()
    logging.basicConfig(level=args.logging_level)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    safe_gpu.claim_gpus()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo, gnn_checkpoint, normalizer = create_models(args)

    tokenizer = BertTokenizerFast.from_pretrained(args.czert)
    czert = BertModel.from_pretrained(args.czert)
    czert = czert.to(device)

    graph_provider = JoinerGraphProvider(tokenizer, czert, device)
    gnn = JoinerGraphModel.from_pretrained(gnn_checkpoint, device)
    gnn.eval()
    gnn = gnn.to(device)

    xml_enhancer = PageXMLEnhancer()

    os.makedirs(args.save, exist_ok=True)

    img_filenames = [img_filename for img_filename in os.listdir(args.img) if os.path.splitext(img_filename)[1] in IMAGE_EXTENSIONS]
    for i, img_filename in enumerate(img_filenames):
        img_extension = os.path.splitext(img_filename)[1]
        xml_filename = img_filename.replace(img_extension, ".xml")
        base_filename = xml_filename.replace(".xml", "")
        json_filename = xml_filename.replace(".xml", ".json")

        img_path = os.path.join(args.img, img_filename)
        xml_path = os.path.join(args.xml, xml_filename)
        alto_path = os.path.join(args.alto, xml_filename) if args.alto is not None else None
        json_save_path = os.path.join(args.save, json_filename)
        xml_save_path = os.path.join(args.save, xml_filename)

        try:
            pagexml = PageLayout(file=xml_path)
        except OSError:
            logging.warning(f"XML file {xml_path} not found. SKIPPING")
            continue

        logging.info(f"({i+1}/{len(img_filenames)}) | Processing: {xml_path}")

        yolo_bites = yolo.produce_bites(img_path, pagexml, alto_path)

        try:
            bites = join_bites(
                yolo_bites,
                gnn,
                graph_provider,
                normalizer,
                base_filename,
                pagexml,
                device,
                CLASSIFICATION_THRESHOLD,
            )
        except RuntimeError:
            logging.info(f"Single region detected on {xml_path}, saving as is.")
            bites = yolo_bites

        try:
            out_xml_string = xml_enhancer.process(pagexml, bites)
            with open(xml_save_path, 'w', encoding='utf-8') as f:
                print(out_xml_string, file=f)
        except UnsupportedLayoutError as e:
            logging.warning(e)

        if args.json:
            save_bites(bites, json_save_path)
    

if __name__ == "__main__":
    main()
