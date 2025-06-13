import nibabel as nib
import numpy as np
from scipy.ndimage import label

# load
vol_nii = nib.load('3702_left_knee.nii.gz')
vol = vol_nii.get_fdata()

# simple bone threshold (Hounsfield > 150 HU)
bone = (vol > 150).astype(np.int32)

# connected components → pick two largest CCs as tibia & femur
labels, num = label(bone)
counts = [(labels==i).sum() for i in range(1, num+1)]
largest2 = np.argsort(counts)[-2:] + 1

mask = np.zeros_like(bone)
mask[labels == largest2[0]] = 1  # tibia
mask[labels == largest2[1]] = 2  # femur

# save
out = nib.Nifti1Image(mask, vol_nii.affine, vol_nii.header)
nib.save(out, 'left_knee_mask.nii.gz')
