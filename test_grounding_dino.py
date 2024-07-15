

from GroundingDINO.groundingdino.util.inference import \
    load_model, load_image, predict, annotate



import os
import supervision as sv
import sys
import torch
from torchvision.ops import box_convert
import matplotlib

home_addr = os.path.expanduser('~') + "/repo/multi-purpose-representation/GroundingDINO"
IMAGE_NAME = "test2.jpg"
CONFIG_PATH = home_addr + "/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = home_addr + "/weights/groundingdino_swint_ogc.pth"
model = load_model(CONFIG_PATH, WEIGHTS_PATH)
print(sys.path)
IMAGE_PATH = home_addr + "/data/" + IMAGE_NAME
print(home_addr)
TEXT_PROMPT = "white chair"
BOX_TRESHOLD = 0.5
TEXT_TRESHOLD = 0.5

image_source, image = load_image(IMAGE_PATH)
print(type(image))
boxes, logits, phrases = predict(
    model=model, 
    image=image, 
    caption=TEXT_PROMPT, 
    box_threshold=BOX_TRESHOLD, 
    text_threshold=TEXT_TRESHOLD
)
print("====Test ===")
print(boxes)
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

sv.plot_image(annotated_frame, (16, 16))
h, w, _ = image_source.shape
print(type(image_source))
boxes = boxes * torch.Tensor([w, h, w, h])
xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
print(boxes)
print(logits)
print(phrases)
print(xyxy)
