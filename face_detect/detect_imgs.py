"""
This code is used to batch detect images in a folder.
"""

import os
import sys
import cv2
import numpy as  np
import torch

from face_detect.vision.ssd.config.fd_config import define_img_size

input_size = 320
test_device = 'cpu'
net_type = 'slim'
threshold = 0.6
candidate_size = 1500

define_img_size(input_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from face_detect.vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from face_detect.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
label_path = os.path.join(model_dir, "voc-model-labels.txt")
test_device = test_device

class_names = [name.strip() for name in open(label_path).readlines()]
if net_type == 'slim':
    model_path = os.path.join(model_dir, "pretrained/version-slim-320.pth")
    # model_path = os.path.join(model_dir, "pretrained/version-slim-640.pth")
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'RFB':
    model_path = os.path.join(model_dir, "pretrained/version-RFB-320.pth")
    # model_path = os.path.join(model_dir, "pretrained/version-RFB-640.pth")
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

fallback_threshold = 0.3

def get_face_boundingbox(orig_image):
    """
        Description:
            In input image, detect face

        Args:
            orig_image: input BGR image.
    """
    rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(rgb, candidate_size / 2, threshold)
    if len(boxes) == 0:
        # partly covered faces (mask, hand) score lower; accept only the single best candidate
        boxes, labels, probs = predictor.predict(rgb, candidate_size / 2, fallback_threshold)
        boxes, probs = boxes[:1], probs[:1]

    if len(boxes) == 0:
        return torch.tensor([]), torch.tensor([])

    # keep faces that touch the image edge, clipped to the image
    height, width, _ = orig_image.shape
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height - 1)
    valid_face = (boxes[:, 2] - boxes[:, 0] > 8) & (boxes[:, 3] - boxes[:, 1] > 8)

    boxes = boxes[valid_face]
    probs = probs[valid_face]

    return boxes, probs
