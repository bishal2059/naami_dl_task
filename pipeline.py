import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import nibabel as nib
import numpy as np
from scipy.spatial.distance import cosine
from torch.utils.data import Dataset

# Utility to ensure tuple parameters
def to_tuple(x):
    return x if isinstance(x, tuple) else (x, x)

# 1. Data loader: split and crop each region to its bounding box
def safe_crop(vol, mask, label, pad=5):
    print(f"Cropping region {label} with padding={pad}")
    idx = np.where(mask == label)
    if len(idx[0]) == 0:
        raise ValueError(f"Mask has no voxels for label {label}")
    bounds = [(a.min(), a.max()) for a in idx]
    slices = []
    for dim, (mn, mx) in enumerate(bounds):
        mn = max(mn - pad, 0)
        mx = min(mx + pad, vol.shape[dim]-1)
        slices.append(slice(mn, mx+1))
    cropped = vol[tuple(slices)]
    norm = (cropped - cropped.mean()) / (cropped.std() + 1e-8)
    print(f"Cropped shape for label {label}: {norm.shape}")
    return norm

class KneeVolumeDataset(Dataset):
    def __init__(self, volume_path, mask_path, pad=5):
        print(f"Loading volume from {volume_path}")
        self.vol = nib.load(volume_path).get_fdata().astype(np.float32)
        print(f"Loading mask from {mask_path}")
        self.mask = nib.load(mask_path).get_fdata().astype(np.int16)
        self.pad = pad

    def __len__(self): return 1

    def __getitem__(self, idx):
        print("Extracting regions...")
        regions = {}
        for label, name in [(1, 'tibia'), (2, 'femur'), (0, 'background')]:
            regions[name] = safe_crop(self.vol, self.mask, label, self.pad)
        return regions

# 2. Recursive 2D->3D converter

def convert_module_2d_to_3d(module, depth=3):
    # print conversion
    cls = module.__class__.__name__
    if isinstance(module, nn.Conv2d):
        print(f"Converting Conv2d to Conv3d: {module.kernel_size} -> depth={depth}")
        w2 = module.weight.data
        b2 = module.bias.data if module.bias is not None else None
        w3 = w2.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth
        conv3d = nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=(depth, *to_tuple(module.kernel_size)),
            stride=(1, *to_tuple(module.stride)),
            padding=(depth//2, *to_tuple(module.padding)),
            bias=(b2 is not None)
        )
        conv3d.weight.data.copy_(w3)
        if b2 is not None: conv3d.bias.data.copy_(b2)
        return conv3d
    if isinstance(module, nn.BatchNorm2d):
        print("Converting BatchNorm2d to BatchNorm3d")
        bn3 = nn.BatchNorm3d(module.num_features)
        bn3.load_state_dict(module.state_dict())
        return bn3
    if isinstance(module, nn.ReLU): return nn.ReLU(module.inplace)
    if isinstance(module, nn.MaxPool2d):
        print("Converting MaxPool2d to MaxPool3d")
        ks, st, pd, dil = to_tuple(module.kernel_size), to_tuple(module.stride), to_tuple(module.padding), to_tuple(module.dilation)
        return nn.MaxPool3d(
            kernel_size=(1, *ks), stride=(1, *st), padding=(0, *pd), dilation=(1, *dil), ceil_mode=module.ceil_mode
        )
    if isinstance(module, nn.AvgPool2d):
        print("Converting AvgPool2d to AvgPool3d")
        ks, st, pd = to_tuple(module.kernel_size), to_tuple(module.stride), to_tuple(module.padding)
        return nn.AvgPool3d(kernel_size=(1, *ks), stride=(1, *st), padding=(0, *pd))
    # recurse
    for name, child in module.named_children():
        setattr(module, name, convert_module_2d_to_3d(child, depth))
    return module

class DenseNet3D(nn.Module):
    def __init__(self, model2d, depth=3):
        super().__init__()
        print("Building DenseNet3D...")
        # collapse conv0
        c0 = model2d.features.conv0
        print("Collapsing conv0 RGB->1 channel")
        w_mean = c0.weight.data.mean(1, keepdim=True)
        r0 = nn.Conv2d(1, c0.out_channels, c0.kernel_size, c0.stride, c0.padding, bias=(c0.bias is not None))
        r0.weight.data.copy_(w_mean)
        if c0.bias is not None: r0.bias.data.copy_(c0.bias.data)
        self.features3d = nn.Sequential()
        self.features3d.add_module('conv0', convert_module_2d_to_3d(r0, depth))
        for name, mod in list(model2d.features.named_children())[1:]:
            self.features3d.add_module(name, convert_module_2d_to_3d(mod, depth))

    def forward(self, x):
        convs = []
        print("Running forward pass through 3D features...")
        for layer in self.features3d:
            x = layer(x)
            if isinstance(layer, nn.Conv3d):
                convs.append(x)
                print(f"Captured feature map from {layer.__class__.__name__} with shape {x.shape}")
        return convs

# 3. Feature extraction with safe indexing
def extract_features(model3d, vol, device):
    print(f"Extracting features on device={device}")
    tensor = torch.from_numpy(vol[None, None]).to(device)
    with torch.no_grad(): outs = model3d(tensor)
    n = len(outs)
    print(f"Total conv layers captured: {n}")
    idxs = [n-1, n-3 if n>=3 else 0, n-5 if n>=5 else 0]
    feats = {}
    for i, idx in enumerate(idxs, 1):
        print(f"Pooling layer-{i} at index {idx}")
        fmap = outs[idx]
        feats[f'layer_{i}'] = F.adaptive_avg_pool3d(fmap, 1).view(-1).cpu().numpy()
    del tensor, outs
    if device.type=='cuda': torch.cuda.empty_cache()
    return feats

# 4. Cosine similarity
def cosine_sim(a, b): return 1 - cosine(a, b)

# 5. Run pipeline
def run_pipeline(vol_path, mask_path, output_csv, device='cpu'):
    print("Starting pipeline...")
    device = torch.device(device)
    regions = KneeVolumeDataset(vol_path, mask_path)[0]
    print("Instantiating pretrained DenseNet121...")
    model2d = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model3d = DenseNet3D(model2d, depth=3).to(device)
    model3d.eval()

    feats = {}
    for name, vol in regions.items():
        print(f"Processing region: {name}")
        feats[name] = extract_features(model3d, vol, device)

    print("Comparing features via cosine similarity...")
    pairs = [('tibia','femur'),('tibia','background'),('femur','background')]
    rows = []
    for a, b in pairs:
        row = {'pair': f'{a}_{b}'}
        for lay in ['layer_1','layer_2','layer_3']:
            sim = cosine_sim(feats[a][lay], feats[b][lay])
            print(f"Cosine({a},{b}) at {lay} = {sim:.4f}")
            row[lay] = sim
        rows.append(row)

    print(f"Saving results to {output_csv}")
    with open(output_csv,'w',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['pair','layer_1','layer_2','layer_3'])
        writer.writeheader()
        writer.writerows(rows)
    print("Pipeline complete.")

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='3D CT feature pipeline')
    parser.add_argument('--volume', required=True)
    parser.add_argument('--mask', required=True)
    parser.add_argument('--output', default='results.csv')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = parser.parse_args()
    run_pipeline(args.volume, args.mask, args.output, args.device)