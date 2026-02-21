from open_clip import create_model_from_pretrained, get_model_config, get_tokenizer
from open_clip.push_to_hf_hub import push_to_hf_hub

model, preprocess = create_model_from_pretrained(
    'ViT-B-16-SigLIP-256',
    # 'ViT-B-32',
    pretrained='/weka/bethge/bkr536/cabs/results/ViT-B-16-SigLIP-256_cabs-fm_recap_0.8/checkpoints/epoch_5.pt',
    load_weights_only=False,
)
# model_config = get_model_config('ViT-B-32')
# tokenizer = get_tokenizer('ViT-B-32')
model_config = get_model_config('ViT-B-16-SigLIP-256')
tokenizer = get_tokenizer('ViT-B-16-SigLIP-256')
push_to_hf_hub(
    model=model,
    tokenizer=tokenizer,
    model_config=model_config,
    repo_id='bethgelab/ViT-B-16-SigLIP-256_cabs-fm_recap_0.8',
)

print('Done!')
