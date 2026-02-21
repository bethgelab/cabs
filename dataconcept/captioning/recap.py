import os
import time
import pickle
import argparse

import torch
from transformers import (
    Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig,
)
from tqdm import tqdm

from dataset import TarDataset, collate_fn
from utils import trim_caption, count_files_pkl, preprocess_caption, clean_and_capitalize, build_prompt

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------------------
# batch inference
# ---------------------------------------------------------------------------

def recaption_batch(model, processor, images, prompts, device):
    messages_batch = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
        for p in prompts
    ]
    texts = processor.apply_chat_template(messages_batch, tokenize=False, add_generation_prompt=True)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            inputs = processor(
                text=texts, images=images,
                return_tensors="pt", padding=True, truncation=True, max_length=400,
            ).to(device, dtype=torch.bfloat16)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)
            del texts, images

            generated_ids = model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=77, do_sample=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                ),
                tokenizer=processor.tokenizer,
            )
        del inputs
        torch.cuda.empty_cache()

    return [
        trim_caption(t)
        for t in processor.batch_decode(generated_ids, skip_special_tokens=True)
    ]


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def load_model(args):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
        device_map=args.device,
        cache_dir=args.cache_dir,
    ).eval()

    for param in model.parameters():
        param.data = param.data.to(torch.bfloat16)
    model.to(args.device, dtype=torch.bfloat16)

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", cache_dir=args.cache_dir,
    )
    return model, processor


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def run_captioning(args):
    os.makedirs(args.results_dir, exist_ok=True)
    base_out = os.path.join(args.results_dir, "qwen_captions", args.pt_dataset)
    os.makedirs(base_out, exist_ok=True)

    model, processor = load_model(args)

    start_num, end_num = int(args.chunk_start), int(args.chunk_end)

    # optional allowlist of chunks to process
    chunks = None
    if args.missing_caps and os.path.exists(args.missing_caps):
        with open(args.missing_caps, "r") as f:
            chunks = set(f.read().splitlines())

    for i in range(start_num, end_num + 1):
        t0 = time.time()
        save_chunk = f"{i:05}"
        pkl_path = os.path.join(base_out, f"{save_chunk}.pkl")

        if chunks is not None and save_chunk not in chunks:
            continue

        chunk_path = os.path.join(args.load_path, f"{save_chunk}.tar")
        if not os.path.exists(chunk_path):
            print(f"not found: {chunk_path}")
            continue

        dataset = TarDataset(chunk_path, transform=None, image_size=args.image_size)

        if os.path.exists(pkl_path) and count_files_pkl(pkl_path) >= len(dataset) - 2:
            print(f"chunk {save_chunk} already done")
            continue

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=4, pin_memory=True,
        )
        print(f"chunk {save_chunk}: {len(dataset)} images")

        recaptions = {}
        with torch.inference_mode(), torch.cuda.amp.autocast():
            for batch in tqdm(dataloader, ascii=True):
                if batch is None:
                    continue

                filenames, images, captions, classes_batch = batch
                captions = [preprocess_caption(c) for c in captions]

                prompts = [
                    build_prompt(captions[idx], classes_batch[idx])
                    for idx in range(len(images))
                ]

                new_captions = recaption_batch(model, processor, images, prompts, args.device)

                recaptions.update({
                    filenames[idx]: clean_and_capitalize(new_captions[idx])
                    for idx in range(len(filenames))
                })
                torch.cuda.empty_cache()

        with open(pkl_path, "wb") as f:
            pickle.dump(recaptions, f, protocol=pickle.HIGHEST_PROTOCOL)

        elapsed = (time.time() - t0) / 60
        print(f"chunk {save_chunk} done in {elapsed:.1f} min\n")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Recaption a tar-based image dataset with Qwen2-VL")
    parser.add_argument("--pt_dataset", type=str, default="datacomp-medium",
                        help="name of the pretraining dataset (used for output folder naming)")
    parser.add_argument("--load_path", type=str, required=True,
                        help="directory containing the .tar shards to recaption")
    parser.add_argument("--chunk_start", type=str, required=True,
                        help="first chunk index to process (zero-padded, e.g. 00100)")
    parser.add_argument("--chunk_end", type=str, required=True,
                        help="last chunk index to process (inclusive)")
    parser.add_argument("--batch_size", type=int, default=450,
                        help="number of images per batch for Qwen2-VL inference")
    parser.add_argument("--image_size", type=int, default=150,
                        help="resize images to this resolution before captioning")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="huggingface cache directory for model weights")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="root directory where caption outputs are saved")
    parser.add_argument("--missing_caps", type=str, default=None,
                        help="optional text file listing chunk ids to process (one per line)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="torch device for inference (e.g. cuda, cuda:0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(f"==> args: {args}")
    run_captioning(args)
