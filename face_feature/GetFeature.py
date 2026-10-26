
import os
import cv2
import numpy as np
import torch
from face_feature.irn50_pytorch import irn50_pytorch
from face_util.faceutil import align_vertical

import ctypes

here = os.path.dirname(os.path.abspath(__file__))
model_feature = irn50_pytorch(os.path.join(here, "irn50_pytorch.npy"))
model_feature.eval()

# optional linear head fine-tuned on LFW (see eval/finetune_head.py), used with its own threshold
tuned_head_path = os.path.join(here, "tuned_head.npz")
tuned_W, tuned_threshold = None, None
if os.path.exists(tuned_head_path):
    with np.load(tuned_head_path) as head:
        tuned_W, tuned_threshold = head["W"], float(head["threshold"])

def apply_tuned_head(feature):
    if tuned_W is None:
        raise FileNotFoundError(tuned_head_path + " missing; run eval/finetune_head.py")
    feature = feature @ tuned_W
    return feature / np.linalg.norm(feature)

def get_face_feature(image, landmark):
    # new buffer per call, otherwise every returned align image is the same array
    feature_align_image = np.zeros([128, 128, 3], dtype=np.uint8)
    landmark_vec = (ctypes.c_float * len(landmark))(*landmark)
    align_vertical(image, image.shape[1], image.shape[0], feature_align_image, 128, 128, 3, landmark_vec, 48, 64, 40)
    # cv2.imwrite("D:/align.png", feature_align_image)
    feature_align_image_proc = feature_align_image / 256
    feature_align_image_proc = torch.from_numpy(feature_align_image_proc.astype(np.float32))
    feature_align_image_proc = feature_align_image_proc.permute(2, 0, 1)
    feature_align_image_proc = feature_align_image_proc.unsqueeze(0)
    feature_out = model_feature(feature_align_image_proc)
    feature_out = torch.nn.functional.normalize(feature_out)[0, :]
    return feature_align_image, feature_out.data.numpy()
