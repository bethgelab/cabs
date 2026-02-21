"""run RAM++ tagging and GroundingDINO detection with multi-scale box ensemble on tar-based image datasets."""

import os
import time
import json
import pickle
import argparse

import torch
import numpy as np
import torchvision
from torch import nn
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from ram.models import ram_plus
from ram import inference_ram_openset as inference
from ram.utils import build_openset_llm_label_embedding

from groundingdino_helpers import load_grounding_model, get_grounding_output
from wbf import bbox_ensemble
from tardataset import TarDataset
from utils import collate_fn, convert_to_rgb, update_json, write_json, count_files_in_tar

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def run_detection(args):
    device = args.device
    detection_sizes = [int(s) for s in args.detection_sizes.split(",")]

    base_out = os.path.join(args.results_dir, "bbox_outputs", args.pt_dataset)
    os.makedirs(base_out, exist_ok=True)

    # load RAM++ and build tag embeddings
    ram_model = ram_plus(pretrained=args.ram_checkpoint, image_size=args.image_size, vit="swin_l")
    print("building tag embeddings...")

    with open(args.class_jsons, "rb") as f:
        descriptions = json.load(f)

    os.makedirs(args.features_dir, exist_ok=True)
    embeddings_path = os.path.join(args.features_dir, "openset_categories.pkl")
    if os.path.exists(embeddings_path):
        cached = pickle.load(open(embeddings_path, "rb"))
        label_embedding, categories = cached[0].to(device), cached[1]
    else:
        label_embedding, categories = build_openset_llm_label_embedding(descriptions, args.cache_dir)
        pickle.dump([label_embedding.cpu(), categories], open(embeddings_path, "wb"))

    tag_list = np.array(categories)
    ram_model.tag_list = tag_list
    ram_model.label_embed = nn.Parameter(label_embedding.float())
    ram_model.num_class = len(categories)
    ram_model.class_threshold = torch.ones(ram_model.num_class) * args.confidence_threshold
    ram_model = ram_model.eval().to(device)

    # load GroundingDINO
    ground_model = load_grounding_model(args.config, args.grounded_checkpoint, device)
    ground_model.to(device)

    # transforms
    default_transform = Compose([convert_to_rgb, ToTensor()])
    tagging_transform = Compose([
        Resize((args.image_size, args.image_size)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    detection_transforms = {
        size: Compose([Resize((size, size)), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        for size in detection_sizes
    }

    start_num, end_num = int(args.chunk_start), int(args.chunk_end)

    # optional allowlist of chunks to process
    chunks = None
    if args.missing_tars and os.path.exists(args.missing_tars):
        with open(args.missing_tars, "r") as f:
            chunks = set(f.read().splitlines())

    for i in range(start_num, end_num + 1):
        t0 = time.time()
        save_chunk = f"{i:05}"
        save_path = os.path.join(base_out, save_chunk)

        if chunks is not None and f"{save_chunk}.tar" not in chunks:
            continue

        chunk_path = os.path.join(args.load_path, f"{save_chunk}.tar")
        if not os.path.exists(chunk_path):
            print(f"not found: {chunk_path}")
            continue

        dataset = TarDataset(chunk_path, transform=default_transform, assume_jsons=True)

        # skip if already processed
        tar_path = save_path + ".tar"
        if os.path.exists(tar_path):
            n = count_files_in_tar(tar_path)
            if n >= len(dataset) - 2:
                print(f"chunk {save_chunk} already done")
                continue

        os.makedirs(save_path, exist_ok=True)

        dataloader = torch.utils.data.DataLoader(
            dataset, collate_fn=collate_fn, batch_size=args.batch_size,
            num_workers=8, shuffle=False, pin_memory=True,
        )
        print(f"chunk {save_chunk}: {len(dataset)} images")

        with torch.inference_mode(), torch.cuda.amp.autocast():
            for batch in tqdm(dataloader, ascii=True):
                if batch is None:
                    continue
                images, json_data_batch, filenames = batch

                # ram++ tagging
                tagging_images = torch.stack([tagging_transform(img) for img in images]).to(device, non_blocking=True)
                ram_res, ram_logits = inference(tagging_images, ram_model, return_logits=True)
                ram_probs = ram_logits.cpu().half() if not ram_logits.is_cuda else ram_logits.half()

                # build captions from top tags sorted by probability
                processed_captions = []
                for idx in range(len(ram_res)):
                    top_k = min(500, ram_res[idx].count(" | ") + 1)
                    top_tags = [tag_list[j] for j in torch.argsort(ram_probs[idx], descending=True)[:top_k]]
                    caption = " ,".join(top_tags).lower().strip()
                    if not caption.endswith("."):
                        caption += "."
                    processed_captions.append(caption)

                # multi-scale grounding dino detection
                images_data = defaultdict(lambda: {
                    "bboxes": [], "scores": [], "labels": [], "tags": None, "tag_probs": None,
                })

                for size, transform in detection_transforms.items():
                    det_images = torch.stack([transform(img) for img in images]).to(device, non_blocking=True)
                    boxes_batch, scores_batch, phrases_batch = get_grounding_output(
                        ground_model, det_images, processed_captions,
                        args.box_threshold, args.text_threshold, device=device,
                    )

                    for idx, fname in enumerate(filenames):
                        if images_data[fname]["tags"] is None:
                            images_data[fname]["tags"] = ram_res[idx]
                            images_data[fname]["tag_probs"] = ram_probs[idx].tolist()

                        # convert center-format boxes to corner-format
                        boxes = boxes_batch[idx].clone()
                        boxes[:, :2] -= boxes[:, 2:] / 2
                        boxes[:, 2:] += boxes[:, :2]

                        # nms per scale
                        nms_idx = torchvision.ops.nms(boxes, scores_batch[idx], args.iou_threshold)
                        boxes = boxes[nms_idx].cpu().detach().half().numpy().tolist()
                        phrases = [phrases_batch[idx][j] for j in nms_idx]
                        scores = scores_batch[idx][nms_idx].cpu().half().numpy().tolist()

                        images_data[fname]["bboxes"].append(boxes)
                        images_data[fname]["scores"].append(scores)
                        images_data[fname]["labels"].append(phrases)

                    del det_images

                # ensemble across scales and write output
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for fname, data in images_data.items():
                        final_bboxes, final_scores, final_labels = bbox_ensemble(
                            data["bboxes"], data["scores"], data["labels"],
                            iou_thr=args.ensemble_threshold,
                        )
                        jd = json_data_batch[filenames.index(fname)]
                        updated = update_json(
                            jd, final_bboxes, final_scores, final_labels,
                            data["tags"], data["tag_probs"],
                        )
                        futures.append(executor.submit(write_json, save_path, fname, updated))
                    for future in futures:
                        future.result()

                del tagging_images, images_data
            torch.cuda.empty_cache()

        elapsed = (time.time() - t0) / 60
        print(f"chunk {save_chunk} done in {elapsed:.1f} min\n")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Run RAM++ tagging + GroundingDINO detection with multi-scale box ensemble",
    )
    parser.add_argument("--pt_dataset", type=str, default="datacomp-medium",
                        help="name of the pretraining dataset (used for output folder naming)")
    parser.add_argument("--load_path", type=str, required=True,
                        help="directory containing the .tar shards to process")
    parser.add_argument("--chunk_start", type=str, required=True,
                        help="first chunk index to process")
    parser.add_argument("--chunk_end", type=str, required=True,
                        help="last chunk index to process (inclusive)")
    parser.add_argument("--class_jsons", type=str, required=True,
                        help="path to the class descriptions JSON (keys=class names, values=descriptions)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="number of images per batch for inference")
    parser.add_argument("--confidence_threshold", type=float, default=0.75,
                        help="RAM++ tag confidence threshold")
    parser.add_argument("--image_size", type=int, default=384,
                        help="image size for RAM++ tagging model input")
    parser.add_argument("--detection_sizes", type=str, default="384,512,800,1000",
                        help="comma-separated image sizes for multi-scale GroundingDINO detection")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="huggingface cache directory for model weights")
    parser.add_argument("--features_dir", type=str, required=True,
                        help="directory to cache openset category embeddings")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="root directory where detection outputs are saved")
    parser.add_argument("--config", type=str, required=True,
                        help="path to GroundingDINO config file (e.g. GroundingDINO_SwinB.py)")
    parser.add_argument("--ram_checkpoint", type=str, required=True,
                        help="path to RAM++ model checkpoint (e.g. ram_plus_swin_large_14m.pth)")
    parser.add_argument("--grounded_checkpoint", type=str, required=True,
                        help="path to GroundingDINO checkpoint (e.g. groundingdino_swinb.pth)")
    parser.add_argument("--box_threshold", type=float, default=0.27,
                        help="GroundingDINO box confidence threshold")
    parser.add_argument("--text_threshold", type=float, default=0.27,
                        help="GroundingDINO text-to-box matching threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.9,
                        help="NMS IoU threshold applied per detection scale")
    parser.add_argument("--ensemble_threshold", type=float, default=0.3,
                        help="weighted box fusion IoU threshold across scales")
    parser.add_argument("--missing_tars", type=str, default=None,
                        help="optional text file listing specific tar filenames to process (one per line)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="torch device for inference (e.g. cuda, cuda:0)")
    return parser.parse_args()


if __name__ == "__main__":
    torchvision.disable_beta_transforms_warning()
    args = get_args()
    print(f"==> args: {args}")
    run_detection(args)
