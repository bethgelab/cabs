import numpy as np
from collections import defaultdict
import random
import warnings
import heapq
import torch.distributed as dist
from open_clip_train.distributed import broadcast_object, all_gather_object
from numba import jit

warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*")


@jit(nopython=True)
def calculate_sample_gain(sample_concepts, concept_counts, target_counts,
                         global_freqs, max_freq):
    total_gain = 0.0
    concept_count = 0
    valid = True

    for i in range(len(sample_concepts)):
        c = sample_concepts[i]
        if c == -1:
            break
            
        if concept_counts[c] >= max_freq:
            return -np.inf, True  # (gain, invalid)
            
        concept_count += 1
        
        if concept_counts[c] < target_counts[c]:
            balance_gain = (target_counts[c] - concept_counts[c]) / max(1, target_counts[c])
            rarity_bonus = 1.0 / max(1.0, global_freqs[c])
            total_gain += balance_gain + rarity_bonus
        else:
            total_gain -= 0.5

    if concept_count == 0:
        return -np.inf, False
    return total_gain / concept_count, False

def diversity_maximisation_curation(args, super_concepts, target_batch_size, 
                                     min_samples_per_concept=1, max_concept_frequency=10):
    # First we need to collect global batch concept representation
    local_sample_concepts = []
    local_concept_to_samples = defaultdict(set)
    
    for idx, concepts in enumerate(super_concepts):
        if isinstance(concepts, (list, tuple)):
            concept_list = list(concepts)
        else:
            concept_list = [concepts]
        
        local_sample_concepts.append(concept_list)
        
        # Concept to sample index mapping
        for concept in concept_list:
            local_concept_to_samples[concept].add(idx)
    
    all_concept_mappings = all_gather_object(args, dict(local_concept_to_samples), dst=0)

    global_concept_to_samples = defaultdict(set)
    sample_offset = 0
    
    if dist.get_rank() == 0:
        for rank, concept_mapping in enumerate(all_concept_mappings):
            for concept, sample_indices in concept_mapping.items():
                global_indices = {idx + sample_offset for idx in sample_indices}
                global_concept_to_samples[concept].update(global_indices)
            sample_offset += len(super_concepts) 
    
    # Broadcast global concept info to all ranks
    if dist.get_rank() == 0:
        # Here, we do concept to sample id mapping
        all_concepts = sorted(global_concept_to_samples.keys())
        concept_to_id = {concept: i for i, concept in enumerate(all_concepts)}
        num_concepts = len(all_concepts)

        base_samples_per_concept = min(target_batch_size // max(1, num_concepts), max_concept_frequency)
        remainder = target_batch_size % max(1, num_concepts)
        
        array_size = num_concepts
        target_concept_counts = np.zeros(array_size, dtype=np.int32)
        global_concept_frequencies = np.zeros(array_size, dtype=np.float32)
        
        for i, concept in enumerate(all_concepts):
            target_count = base_samples_per_concept
            if i < remainder and base_samples_per_concept < max_concept_frequency:
                target_count += 1
            target_concept_counts[i] = max(min(target_count, max_concept_frequency), 
                                      min_samples_per_concept)
            global_concept_frequencies[i] = len(global_concept_to_samples[concept])

        total_target = np.sum(target_concept_counts)
        if total_target > target_batch_size:
            scale_factor = target_batch_size / total_target
            target_concept_counts = np.maximum(1, 
                                            np.minimum((target_concept_counts * scale_factor).astype(np.int32),
                                            max_concept_frequency))
        
        # Need to broadcats this data to all ranks
        broadcast_data = {
            'concept_to_id': concept_to_id,
            'target_counts': target_concept_counts,
            'global_freqs': global_concept_frequencies
        }
    else:
        broadcast_data = None
    broadcast_data = broadcast_object(args,broadcast_data, src=0)
    concept_to_id = broadcast_data['concept_to_id']
    target_concept_counts = broadcast_data['target_counts']
    global_concept_frequencies = broadcast_data['global_freqs']
    num_concepts = len(concept_to_id)
    
    # Local mapping
    max_concepts_per_sample = max(len(concepts) for concepts in local_sample_concepts) if local_sample_concepts else 0
    sample_concepts_array = np.full((len(local_sample_concepts), max_concepts_per_sample), -1, dtype=np.int32)
    
    for idx, concepts in enumerate(local_sample_concepts):
        for i, concept in enumerate(concepts):
            sample_concepts_array[idx, i] = concept_to_id[concept]
    
    # Inverted index for concepts to samples
    inverted_index = defaultdict(list)
    for idx in range(len(local_sample_concepts)):
        for j in range(max_concepts_per_sample):
            cid = sample_concepts_array[idx, j]
            if cid == -1:
                break
            inverted_index[cid].append(idx)
    
    concept_counts = np.zeros(num_concepts, dtype=np.int32)
    valid_mask = np.ones(len(local_sample_concepts), dtype=bool)
    current_gains = np.full(len(local_sample_concepts), -np.inf, dtype=np.float32)
    
    heap = []
    for idx in range(len(local_sample_concepts)):
        concepts = sample_concepts_array[idx]
        gain, invalid = calculate_sample_gain(
            concepts, concept_counts, target_concept_counts,
            global_concept_frequencies, max_concept_frequency
        )
        current_gains[idx] = gain
        if not invalid and gain > -np.inf:
            heapq.heappush(heap, (-gain, idx))
    
    my_target_count = target_batch_size // args.world_size
    if dist.get_rank() < target_batch_size % args.world_size:
        my_target_count += 1
    
    selected_samples = []

    for _ in range(min(my_target_count, len(local_sample_concepts))):
        if not heap:
            break
            
        while heap:
            neg_gain, idx = heapq.heappop(heap)
            if valid_mask[idx] and -neg_gain == current_gains[idx]:
                selected_samples.append(idx)
                valid_mask[idx] = False
                break
        for j in range(max_concepts_per_sample):
            cid = sample_concepts_array[idx, j]
            if cid == -1:
                break
            concept_counts[cid] += 1
            
            # Invalidate samples when concept reaches max frequency
            if concept_counts[cid] >= max_concept_frequency:
                for neighbor in inverted_index[cid]:
                    if valid_mask[neighbor]:
                        valid_mask[neighbor] = False
                        current_gains[neighbor] = -np.inf
            else:
                for neighbor in inverted_index[cid]:
                    if not valid_mask[neighbor] or neighbor == idx:
                        continue
                    
                    new_gain, invalid = calculate_sample_gain(
                        sample_concepts_array[neighbor], concept_counts, 
                        target_concept_counts, global_concept_frequencies,
                        max_concept_frequency
                    )
                    if invalid or new_gain == -np.inf:
                        valid_mask[neighbor] = False
                        current_gains[neighbor] = -np.inf
                    elif new_gain != current_gains[neighbor]:
                        current_gains[neighbor] = new_gain
                        heapq.heappush(heap, (-new_gain, neighbor))
    
    # If we still need more samples, fill randomly from valid pool
    if len(selected_samples) < my_target_count:
        available_indices = np.where(valid_mask)[0]
        additional_needed = my_target_count - len(selected_samples)
        
        if len(available_indices) > 0:
            additional_samples = np.random.choice(
                available_indices, 
                size=min(additional_needed, len(available_indices)),
                replace=False
            )
            selected_samples.extend(additional_samples.tolist())
    
    final_indices = np.array(selected_samples, dtype=np.int64)
    final_indices = final_indices[final_indices < len(local_sample_concepts)]  # Safety check
    
    return final_indices


def apply_dm_curation(args, super_classes, target_batch_size, 
                           min_samples_per_class=1,
                           max_concept_frequency=10):
    return diversity_maximisation_curation(
            args=args,
            super_concepts=super_classes,
            target_batch_size=target_batch_size,
            min_samples_per_concept=min_samples_per_class,
            max_concept_frequency=max_concept_frequency
        )