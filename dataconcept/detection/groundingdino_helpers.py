"""GroundingDINO model loading and batch inference helpers."""

import torch

from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict, get_phrases_from_posmap, get_closest_class,
)


def load_grounding_model(config_path, checkpoint_path, device="cuda"):
    """load GroundingDINO from config and checkpoint."""
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    return model.eval()


def get_grounding_output(model, images, captions, box_threshold, text_threshold, device="cuda"):
    """batch GroundingDINO inference: returns filtered boxes, scores, and class labels per image."""
    model = model.to(device)
    images = images.to(device)

    with torch.inference_mode(), torch.cuda.amp.autocast():
        outputs = model(images, captions=captions)

    logits_batch = outputs["pred_logits"].sigmoid()
    boxes_batch = outputs["pred_boxes"]

    tokenizer = model.tokenizer
    tokenized_captions = [tokenizer(c) for c in captions]

    all_boxes, all_scores, all_phrases = [], [], []

    for caption, tokenized, logits, boxes in zip(captions, tokenized_captions, logits_batch, boxes_batch):
        correct_preds = [cls.strip() for cls in caption.removesuffix(".").split(" ,")]

        # filter by box threshold
        mask = logits.max(dim=1)[0] > box_threshold
        logits_filt = logits[mask].cpu()
        boxes_filt = boxes[mask].cpu()

        phrases, scores, kept_boxes = [], [], []
        for logit, box in zip(logits_filt, boxes_filt):
            pred = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            closest = get_closest_class(pred, correct_preds)
            if not closest:
                continue
            kept_boxes.append(box)
            phrases.append(closest)
            scores.append(logit.max().item())

        all_boxes.append(torch.stack(kept_boxes) if kept_boxes else torch.zeros(0, 4))
        all_scores.append(torch.tensor(scores))
        all_phrases.append(phrases)

    return all_boxes, all_scores, all_phrases
