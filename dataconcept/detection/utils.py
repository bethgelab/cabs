"""shared utilities for the detection pipeline."""

import os
import json
import tarfile

import numpy as np
import torch


def collate_fn(batch):
    """filter out failed samples and group by field."""
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    images, jsons, filepaths = zip(*valid)
    return list(images), list(jsons), list(filepaths)


def convert_to_rgb(image):
    return image.convert("RGB")


def count_files_in_tar(tar_path):
    """count regular files in a tar archive."""
    with tarfile.open(tar_path, "r") as tar:
        return sum(1 for m in tar.getmembers() if m.isfile())


def update_json(json_data, bboxes, scores, labels, tags, tag_probs):
    """add detection results to the sample's json metadata."""
    def to_list(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        return obj

    json_data["bounding_boxes"] = to_list(bboxes)
    json_data["scores"] = to_list(scores)
    json_data["classes"] = to_list(labels)
    json_data["tags"] = to_list(tags)
    json_data["tag_probs"] = to_list(tag_probs)
    return json_data


def write_json(save_dir, filename, data):
    """write a json file with compact array formatting."""
    json_str = json.dumps(data, indent=1, separators=(",", ": "))
    # flatten arrays onto single lines for readability
    json_str = json_str.replace("[\n", "[").replace("\n]", "]").replace("\n  ", " ")
    path = os.path.join(save_dir, os.path.splitext(filename)[0] + ".json")
    with open(path, "w") as f:
        f.write(json_str)
