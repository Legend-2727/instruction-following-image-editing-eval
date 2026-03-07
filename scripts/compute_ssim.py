#!/usr/bin/env python
"""Compute SSIM between source and target images for test set."""
import json, numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim

data_dir = Path('data/magicbrush')
records = [json.loads(l) for l in open('runs/classifier/test_set.jsonl')]

ssim_vals = []
for i, rec in enumerate(records):
    src = np.array(Image.open(data_dir / rec['source_path']).convert('RGB').resize((256,256)))
    tgt = np.array(Image.open(data_dir / rec['target_path']).convert('RGB').resize((256,256)))
    val = ssim(src, tgt, channel_axis=2)
    ssim_vals.append(val)
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(records)}")

print(f'SSIM: mean={np.mean(ssim_vals):.4f} std={np.std(ssim_vals):.4f} median={np.median(ssim_vals):.4f} n={len(ssim_vals)}')

with open('runs/benchmark/benchmark_results.json') as f:
    results = json.load(f)
results['ssim'] = {
    'ssim_mean': float(np.mean(ssim_vals)),
    'ssim_std': float(np.std(ssim_vals)),
    'ssim_median': float(np.median(ssim_vals)),
    'n_samples': len(ssim_vals),
}
with open('runs/benchmark/benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('Updated benchmark_results.json')
