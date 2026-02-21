"""run RAM++ open-set tagging on tar-based image datasets, saving tags and probabilities as pickle files."""

import os
import time
import json
import pickle
import argparse

import torch
import numpy as np
from torch import nn
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from tqdm import tqdm

from ram.models import ram_plus
from ram import inference_ram_openset as inference
from ram.utils import build_openset_llm_label_embedding
from tardataset import TarDataset
from utils import collate_fn, convert_to_rgb


def run_tagging(args):
    device = args.device
    print(f"building model on: {device}")

    model = ram_plus(pretrained=args.ram_checkpoint, image_size=args.image_size, vit="swin_l")
    print("building tag embeddings...")

    with open(args.class_jsons, "rb") as f:
        descriptions = json.load(f)

    os.makedirs(args.features_dir, exist_ok=True)
    embeddings_path = os.path.join(args.features_dir, "rampp_categories.pkl")
    if os.path.exists(embeddings_path):
        cached = pickle.load(open(embeddings_path, "rb"))
        label_embedding, categories = cached[0].to(device), cached[1]
    else:
        label_embedding, categories = build_openset_llm_label_embedding(descriptions, args.cache_dir)
        pickle.dump([label_embedding.cpu(), categories], open(embeddings_path, "wb"))

    model.tag_list = np.array(categories)
    model.label_embed = nn.Parameter(label_embedding.float())
    model.num_class = len(categories)
    model.class_threshold = torch.ones(model.num_class) * args.confidence_threshold
    model = model.eval().to(device)

    transforms = Compose([
        convert_to_rgb,
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_out = os.path.join(args.results_dir, "rampp_outputs", args.pt_dataset)
    os.makedirs(base_out, exist_ok=True)

    start_num, end_num = int(args.chunk_start), int(args.chunk_end)

    for i in range(start_num, end_num + 1):
        t0 = time.time()
        save_chunk = f"{i:06}"
        pkl_path = os.path.join(base_out, f"rampp_output_{args.pt_dataset}_{save_chunk}.pkl")

        chunk_path = os.path.join(args.load_path, f"{save_chunk}.tar")
        if not os.path.exists(chunk_path):
            print(f"not found: {chunk_path}")
            continue

        dataset = TarDataset(chunk_path, transform=transforms, assume_jsons=False)
        dataloader = torch.utils.data.DataLoader(
            dataset, collate_fn=collate_fn, batch_size=args.batch_size,
            num_workers=4, shuffle=False, pin_memory=True,
        )
        print(f"chunk {save_chunk}: {len(dataset)} images")

        outs = {"confidence_threshold": args.confidence_threshold, "filenames": [], "tags": [], "probs": []}

        with torch.inference_mode(), torch.cuda.amp.autocast():
            for batch in tqdm(dataloader, ascii=True):
                if batch is None:
                    continue
                images, _, filenames = batch

                if isinstance(images, list):
                    images = torch.stack(images, dim=0).to(device, non_blocking=True)
                else:
                    images = images.to(device, non_blocking=True)

                res, logits = inference(images, model, return_logits=True)
                probs = logits.half().cpu().numpy()

                for f, tag, prob in zip(filenames, res, probs):
                    outs["filenames"].append(f)
                    outs["tags"].append(tag)
                    outs["probs"].append(prob)

        with open(pkl_path, "wb") as f:
            pickle.dump(outs, f, protocol=pickle.HIGHEST_PROTOCOL)

        elapsed = (time.time() - t0) / 60
        print(f"chunk {save_chunk} done in {elapsed:.1f} min\n")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Run RAM++ open-set tagging on tar-based image datasets")
    parser.add_argument("--pt_dataset", type=str, default="datacomp-medium",
                        help="name of the pretraining dataset (used for output folder naming)")
    parser.add_argument("--load_path", type=str, required=True,
                        help="directory containing the .tar shards to process")
    parser.add_argument("--chunk_start", type=str, required=True,
                        help="first chunk index to process (zero-padded, e.g. 000000)")
    parser.add_argument("--chunk_end", type=str, required=True,
                        help="last chunk index to process (inclusive)")
    parser.add_argument("--class_jsons", type=str, required=True,
                        help="path to the class descriptions JSON file")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="number of images per batch for inference")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="RAM++ tag confidence threshold")
    parser.add_argument("--ram_checkpoint", type=str, required=True,
                        help="path to RAM++ model checkpoint (e.g. ram_plus_swin_large_14m.pth)")
    parser.add_argument("--image_size", type=int, default=384,
                        help="image size for RAM++ model input")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="huggingface cache directory for model weights")
    parser.add_argument("--features_dir", type=str, required=True,
                        help="directory to cache openset category embeddings")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="root directory where tagging outputs are saved")
    parser.add_argument("--device", type=str, default="cuda",
                        help="torch device for inference (e.g. cuda, cuda:0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(f"==> args: {args}")
    run_tagging(args)
