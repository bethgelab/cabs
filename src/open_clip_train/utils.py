import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

import torch
import numpy as np
import open_clip

def analyze_class_distribution(accumulated_classes):
    """
    Create three histogram plots analyzing class distribution
    
    Args:
        accumulated_classes: List of lists of strings (each inner list is classes for one sample)
    """
    
    # Flatten all classes for overall distribution
    all_classes_flat = [cls for sample_classes in accumulated_classes for cls in sample_classes]
    
    # Count all classes
    all_class_counts = Counter(all_classes_flat)
    unique_classes_count = len(all_class_counts)

    prefix = 'results/gibbs_batch/maxent_20_'
    
    print(f"Total number of unique classes: {unique_classes_count}")
    print(f"Total number of class instances: {len(all_classes_flat)}")
    print(f"Total number of samples: {len(accumulated_classes)}")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: All classes
    classes_all = list(all_class_counts.keys())
    counts_all = list(all_class_counts.values())
    
    axes[0].bar(range(len(classes_all)), counts_all)
    axes[0].set_title(f'All Classes Distribution\n({unique_classes_count} unique classes)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Class Index')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Top 50 classes
    top_50_items = all_class_counts.most_common(50)
    top_50_classes = [item[0] for item in top_50_items]
    top_50_counts = [item[1] for item in top_50_items]
    
    axes[1].bar(range(len(top_50_classes)), top_50_counts)
    axes[1].set_title(f'Top 50 Most Frequent Classes\n(out of {unique_classes_count} unique classes)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Class Rank')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xticks(range(0, len(top_50_classes), 5))
    axes[1].set_xticklabels([f'{i+1}' for i in range(0, len(top_50_classes), 5)])
    axes[1].grid(True, alpha=0.3)
    
    # Add class names as labels for top 10
    for i in range(min(10, len(top_50_classes))):
        axes[1].text(i, top_50_counts[i] + max(top_50_counts) * 0.01, 
                    top_50_classes[i], rotation=45, ha='left', va='bottom', fontsize=8)
    
    # Plot 3: Unique classes per sample (deduplicated within each sample)
    unique_per_sample = []
    for sample_classes in accumulated_classes:
        # Remove duplicates within each sample while preserving order
        seen = set()
        unique_sample_classes = []
        for cls in sample_classes:
            if cls not in seen:
                seen.add(cls)
                unique_sample_classes.append(cls)
        unique_per_sample.extend(unique_sample_classes)
    
    unique_sample_counts = Counter(unique_per_sample)
    unique_sample_classes_count = len(unique_sample_counts)
    
    # Get top 50 for the unique per sample distribution
    top_50_unique_items = unique_sample_counts.most_common(50)
    top_50_unique_classes = [item[0] for item in top_50_unique_items]
    top_50_unique_counts = [item[1] for item in top_50_unique_items]
    
    axes[2].bar(range(len(top_50_unique_classes)), top_50_unique_counts)
    axes[2].set_title(f'Top 50 Classes (Unique per Sample)\n({unique_sample_classes_count} unique classes, {len(unique_per_sample)} total instances)', 
                     fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Class Rank')
    axes[2].set_ylabel('Frequency')
    axes[2].set_xticks(range(0, len(top_50_unique_classes), 5))
    axes[2].set_xticklabels([f'{i+1}' for i in range(0, len(top_50_unique_classes), 5)])
    axes[2].grid(True, alpha=0.3)
    
    # Add class names as labels for top 10
    for i in range(min(10, len(top_50_unique_classes))):
        axes[2].text(i, top_50_unique_counts[i] + max(top_50_unique_counts) * 0.01, 
                    top_50_unique_classes[i], rotation=45, ha='left', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the combined figure
    # plt.savefig('results/gibbs_batch/class_distribution_analysis.png', dpi=300, bbox_inches='tight')
    # plt.savefig('results/gibbs_batch/class_distribution_analysis.pdf', bbox_inches='tight')
    # print("Saved combined plot: class_distribution_analysis.png and .pdf")
    
    plt.show()
    
    # Save individual plots with higher resolution
    fig_individual = plt.figure(figsize=(15, 12))
    
    # Individual plot 1: All classes
    plt.subplot(3, 1, 1)
    plt.bar(range(len(classes_all)), counts_all)
    plt.title(f'All Classes Distribution ({unique_classes_count} unique classes)', fontsize=14, fontweight='bold')
    plt.xlabel('Class Index')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Individual plot 2: Top 50 classes  
    plt.subplot(3, 1, 2)
    plt.bar(range(len(top_50_classes)), top_50_counts)
    plt.title(f'Top 50 Most Frequent Classes (out of {unique_classes_count} unique classes)', fontsize=14, fontweight='bold')
    plt.xlabel('Class Rank')
    plt.ylabel('Frequency')
    plt.xticks(range(0, len(top_50_classes), 5), [f'{i+1}' for i in range(0, len(top_50_classes), 5)])
    plt.grid(True, alpha=0.3)
    
    # Add class names for top 10
    for i in range(min(10, len(top_50_classes))):
        plt.text(i, top_50_counts[i] + max(top_50_counts) * 0.01, 
                top_50_classes[i], rotation=45, ha='left', va='bottom', fontsize=10)
    
    # Individual plot 3: Unique per sample
    plt.subplot(3, 1, 3)
    plt.bar(range(len(top_50_unique_classes)), top_50_unique_counts)
    plt.title(f'Top 50 Classes - Unique per Sample ({unique_sample_classes_count} unique classes, {len(unique_per_sample)} total instances)', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Class Rank')
    plt.ylabel('Frequency')
    plt.xticks(range(0, len(top_50_unique_classes), 5), [f'{i+1}' for i in range(0, len(top_50_unique_classes), 5)])
    plt.grid(True, alpha=0.3)
    
    # Add class names for top 10
    for i in range(min(10, len(top_50_unique_classes))):
        plt.text(i, top_50_unique_counts[i] + max(top_50_unique_counts) * 0.01, 
                top_50_unique_classes[i], rotation=45, ha='left', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{prefix}class_distribution_detailed.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{prefix}class_distribution_detailed.pdf', bbox_inches='tight')
    print("Saved detailed plot: class_distribution_detailed.png and .pdf")
    plt.show()
    
    # Save separate individual plots
    # Plot 1: All classes distribution
    # plt.figure(figsize=(12, 6))
    # plt.bar(range(len(classes_all)), counts_all)
    # plt.title(f'All Classes Distribution\n{unique_classes_count} unique classes, {len(all_classes_flat)} total instances', 
    #           fontsize=16, fontweight='bold')
    # plt.xlabel('Class Index', fontsize=12)
    # plt.ylabel('Frequency', fontsize=12)
    # plt.grid(True, alpha=0.3)
    # plt.savefig('results/gibbs_batch/plot1_all_classes.png', dpi=300, bbox_inches='tight')
    # plt.savefig('results/gibbs_batch/plot1_all_classes.pdf', bbox_inches='tight')
    # print("Saved: plot1_all_classes.png and .pdf")
    # plt.show()
    
    # # Plot 2: Top 50 classes
    # plt.figure(figsize=(12, 6))
    # plt.bar(range(len(top_50_classes)), top_50_counts)
    # plt.title(f'Top 50 Most Frequent Classes\nOut of {unique_classes_count} unique classes', 
    #           fontsize=16, fontweight='bold')
    # plt.xlabel('Class Rank', fontsize=12)
    # plt.ylabel('Frequency', fontsize=12)
    # plt.xticks(range(0, len(top_50_classes), 5), [f'{i+1}' for i in range(0, len(top_50_classes), 5)])
    # plt.grid(True, alpha=0.3)
    
    # # Add top 15 class names
    # for i in range(min(15, len(top_50_classes))):
    #     plt.text(i, top_50_counts[i] + max(top_50_counts) * 0.01, 
    #             top_50_classes[i], rotation=45, ha='left', va='bottom', fontsize=9)
    
    # plt.savefig('results/gibbs_batch/plot2_top50_classes.png', dpi=300, bbox_inches='tight')
    # plt.savefig('results/gibbs_batch/plot2_top50_classes.pdf', bbox_inches='tight')
    # print("Saved: plot2_top50_classes.png and .pdf")
    # plt.show()
    
    # # Plot 3: Unique per sample
    # plt.figure(figsize=(12, 6))
    # plt.bar(range(len(top_50_unique_classes)), top_50_unique_counts)
    # plt.title(f'Top 50 Classes (Unique per Sample)\n{unique_sample_classes_count} unique classes, {len(unique_per_sample)} total instances', 
    #           fontsize=16, fontweight='bold')
    # plt.xlabel('Class Rank', fontsize=12)
    # plt.ylabel('Frequency (Number of Samples)', fontsize=12)
    # plt.xticks(range(0, len(top_50_unique_classes), 5), [f'{i+1}' for i in range(0, len(top_50_unique_classes), 5)])
    # plt.grid(True, alpha=0.3)
    
    # # Add top 15 class names
    # for i in range(min(15, len(top_50_unique_classes))):
    #     plt.text(i, top_50_unique_counts[i] + max(top_50_unique_counts) * 0.01, 
    #             top_50_unique_classes[i], rotation=45, ha='left', va='bottom', fontsize=9)
    
    # plt.savefig('results/gibbs_batch/plot3_unique_per_sample.png', dpi=300, bbox_inches='tight')
    # plt.savefig('results/gibbs_batch/plot3_unique_per_sample.pdf', bbox_inches='tight')
    # print("Saved: plot3_unique_per_sample.png and .pdf")
    # plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\n1. ALL CLASSES:")
    print(f"   - Unique classes: {unique_classes_count}")
    print(f"   - Total instances: {len(all_classes_flat)}")
    print(f"   - Average instances per class: {len(all_classes_flat)/unique_classes_count:.2f}")
    print(f"   - Most frequent class: '{top_50_items[0][0]}' ({top_50_items[0][1]} times)")
    
    print(f"\n2. UNIQUE PER SAMPLE:")
    print(f"   - Unique classes: {unique_sample_classes_count}")  
    print(f"   - Total instances: {len(unique_per_sample)}")
    print(f"   - Average instances per class: {len(unique_per_sample)/unique_sample_classes_count:.2f}")
    print(f"   - Most frequent class: '{top_50_unique_items[0][0]}' ({top_50_unique_items[0][1]} samples)")
    
    print(f"\n3. PER SAMPLE STATISTICS:")
    sample_lengths = [len(sample) for sample in accumulated_classes]
    unique_sample_lengths = [len(set(sample)) for sample in accumulated_classes]
    
    print(f"   - Average classes per sample (with duplicates): {np.mean(sample_lengths):.2f}")
    print(f"   - Average unique classes per sample: {np.mean(unique_sample_lengths):.2f}")
    print(f"   - Max classes in a sample: {max(sample_lengths)}")
    print(f"   - Min classes in a sample: {min(sample_lengths)}")
    
    return all_class_counts, unique_sample_counts

# Usage:
# all_counts, unique_counts = analyze_class_distribution(accumulated_classes)



def verify_model_loaded(model, preprocess_val, device):
    print("\n=== OpenCLIP Pretrained Model Verification ===")

    # 1️⃣ Parameter statistics
    with torch.no_grad():
        all_params = torch.cat([p.flatten() for p in model.parameters() if p.requires_grad])
    print(f"[Param Stats] mean={all_params.mean().item():.6f}, std={all_params.std().item():.6f}")
    print(f"[Param Range] min={all_params.min().item():.6f}, max={all_params.max().item():.6f}")

    # 2️⃣ First conv weight fingerprint
    conv1_w = model.visual.conv1.weight.detach().cpu().numpy().flatten()
    print(f"[Conv1 slice] {conv1_w[:5]}")

    # 3️⃣ Quick forward pass sanity check
    try:
        from PIL import Image
        import requests
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Black_square.jpg/1200px-Black_square.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        image_tensor = preprocess_val(image).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.encode_image(image_tensor)
        print(f"[Forward Pass] embedding[:5] = {emb[0, :5].cpu().numpy()}")
    except Exception as e:
        print("[Forward Pass] Skipped (error loading test image)", e)

    # 4️⃣ Missing/unexpected keys check — only works if we have a local checkpoint path
    if hasattr(model, "_is_pretrained") and model._is_pretrained:
        print("[State Dict] Model reports pretrained=True")
    else:
        print("[State Dict] WARNING: Model may be randomly initialized!")

    print("=== Verification Done ===\n")
