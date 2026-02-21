"""
Sampling strategies for CABS.

This module contains various sampling functions used for batch curation:
- cabs_freq_sampling: Select samples with most concepts
"""

import numpy as np
from collections import Counter, defaultdict
import random


def cabs_freq_sampling(super_classes, target):
    """
    Select samples based on concept frequency (number of concepts per sample).

    Selects samples that have the most unique concepts, favoring samples
    with rich concept coverage.

    Args:
        super_classes: List of concept lists for each sample
        target: Number of samples to select

    Returns:
        np.ndarray: Indices of selected samples
    """
    pairs = [(len(cls_list), i) for i, cls_list in enumerate(super_classes)]
    pairs.sort(key=lambda x: -x[0])
    return np.array([idx for _, idx in pairs[:target]], dtype=np.int64)
