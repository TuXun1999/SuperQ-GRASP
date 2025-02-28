from GroundingDINO.groundingdino.util.inference import \
    load_model, load_image, predict, annotate

from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamPredictor
import numpy as np

import os
import supervision as sv
import sys
import torch
from torchvision.ops import box_convert
import matplotlib
import cv2
from PIL import Image

home_addr = os.path.expanduser('~') + "/repo/multi-purpose-representation/GroundingDINO"
IMAGE_NAME = "pose_estimation.jpg"
CONFIG_PATH = home_addr + "/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = home_addr + "/weights/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT_PATH = home_addr + "/sam_weights/sam_vit_h_4b8939.pth"
SAM_ENCODER_VERSION = "vit_h"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(CONFIG_PATH, WEIGHTS_PATH, DEVICE)
print(sys.path)
IMAGE_PATH = home_addr + "/data/" + IMAGE_NAME
print(home_addr)
TEXT_PROMPT = "blue chair"
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.3

image_source, image = load_image(IMAGE_PATH)
print(type(image))
boxes, logits, phrases = predict(
    model=model, 
    image=image, 
    caption=TEXT_PROMPT, 
    box_threshold=BOX_TRESHOLD, 
    text_threshold=TEXT_TRESHOLD
)
# print("====Test ===")
# print(boxes)
# annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

# sv.plot_image(annotated_frame, (16, 16))
h, w, _ = image_source.shape
print(type(image_source))
boxes = boxes * torch.Tensor([w, h, w, h])
xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
print(boxes)
print(logits)
print(phrases)
print(xyxy)



sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

confidence = logits.numpy()
detections = sv.Detections(xyxy=xyxy, confidence=confidence)
detections.class_id = np.array([0, 1])
# convert detections to masks
detections.mask = segment(
    sam_predictor=sam_predictor,
    image=image_source,
    xyxy=xyxy
)

image_masked = np.zeros((h, w, 4), dtype=np.uint8)
print("===== Mask ====")
print(detections.mask.shape)
print(image_source.shape)
mask = detections.mask
image_source = np.array(image_source)
for i in range(h):
    for j in range(w):
        if mask[0][i][j]:
            image_masked[i, j] = np.array(\
                [image_source[i, j, 0], image_source[i, j, 1],\
                 image_source[i, j, 2], 255])

print(image_masked[442, 213])
# Save the new masked image
data = Image.fromarray(image_masked, 'RGBA') 
data.save('test_masked.png') 
# Annotate image with detections
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()
labels = [TEXT_PROMPT + " " + str(logits[0].numpy())]
annotated_image = mask_annotator.annotate(scene=image_source.copy(), detections=detections)
#annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# matplotlib inline
sv.plot_image(annotated_image, (16, 16))