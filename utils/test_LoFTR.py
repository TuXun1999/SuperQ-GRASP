import torch
import cv2
import numpy as np
import matplotlib.cm as cm
import sys
import os
sys.path.append(os.getcwd())
from LoFTR.src.utils.plotting import make_matching_figure
from LoFTR.src.loftr import LoFTR, default_cfg


img0_pth = "./LoFTR/0010.jpg"
img1_pth = "./LoFTR/hand_color_image_0003.jpg"
image_pair = [img0_pth, img1_pth]
image_type = "outdoor"

# Build up the feature matcher
matcher = LoFTR(config=default_cfg)
if image_type == 'indoor':
  matcher.load_state_dict(torch.load("./LoFTR/weights/indoor_ds_new.ckpt")['state_dict'])
elif image_type == 'outdoor':
  matcher.load_state_dict(torch.load("./LoFTR/weights/outdoor_ds.ckpt")['state_dict'])
else:
  raise ValueError("Wrong image_type is given.")
matcher = matcher.eval().cuda()

# Read the images & Preprocessing
img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw, (640, 480))
img1_raw = cv2.resize(img1_raw, (640, 480))

img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}

# Inference with LoFTR and get prediction
with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()

idx = mconf > 0.5
mkpts0 = mkpts0[idx]
mkpts1 = mkpts1[idx]
mconf = mconf[idx]

img0_original = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
img1_original = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
h0, w0 = img0_original.shape
h1, w1 = img1_original.shape
print(h0)
print(w0)
mkpts0[:, 0] = mkpts0[:, 0] * (w0/640)
mkpts0[:, 1] = mkpts0[:, 1] * (h0/480)
mkpts1[:, 0] = mkpts1[:, 0] * (w1/640)
mkpts1[:, 1] = mkpts1[:, 1] * (h1/480)

# Draw 
color = cm.jet(mconf, alpha=0.7)
text = [
    'LoFTR',
    'Matches: {}'.format(len(mkpts0)),
]
make_matching_figure(img0_original, img1_original, mkpts0, mkpts1, color, mkpts0, mkpts1, text,\
                     path="LoFTR-demo.jpg")
