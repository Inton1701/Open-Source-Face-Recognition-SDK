
import argparse
import cv2
import torch
import numpy as np
import ctypes
import os.path
import time

from face_detect.detect_imgs import get_face_boundingbox
from face_landmark.GetLandmark import get_face_landmark
from face_feature.GetFeature import get_face_feature, apply_tuned_head, tuned_threshold
from face_pose.GetPose import get_face_pose

threshold = 75

def to_bgr(image):
    """Validate an input image and return it as a contiguous uint8 BGR array."""
    if image is None:
        raise ValueError('image is None (file missing or not a readable image)')
    if not isinstance(image, np.ndarray):
        raise TypeError('image must be a numpy array, got %s' % type(image).__name__)
    if image.size == 0:
        raise ValueError('image is empty')
    if image.dtype != np.uint8:
        raise ValueError('image must be uint8 (0-255), got %s' % image.dtype)
    image = np.ascontiguousarray(image)
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError('image must be BGR, BGRA or grayscale, got shape %s' % (image.shape,))

def GetImageInfo(image, faceMaxCount, tuned=False):
    """tuned=True returns features from the LFW fine-tuned head; compare them against tuned_threshold."""
    image = to_bgr(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ### Detection
    start_time = time.time() * 1000
    boxes, scores = get_face_boundingbox(image)
    boxes = boxes[:faceMaxCount]
    scores = scores[:faceMaxCount]
    count = len(boxes)
    bboxes = []
    bscores = []
    for idx in range(count):
        bboxes.append(boxes[idx].data.numpy())
        bscores.append(scores[idx].data.numpy())

    ### Landmark
    start_time = time.time() * 1000
    landmarks = [] ### np.zeros((count, 136), dtype=np.float32)
    for idx in range(count):
        landmarks.append(get_face_landmark(gray_image, boxes[idx]).data.numpy())

    ### Pose
    poses = []
    for idx in range(count):
        poses.append(get_face_pose(boxes[idx], landmarks[idx]))

    ### Feature
    start_time = time.time() * 1000
    features = []
    alignimgs = []
    for idx in range(count):
        alignimg, feature = get_face_feature(image, landmarks[idx])
        features.append(apply_tuned_head(feature) if tuned else feature)
        alignimgs.append(alignimg)
    # print("Feature extraction time = %s ms" % (time.time() * 1000 - start_time))

    return count, bboxes, bscores, landmarks, alignimgs, features

def get_similarity(feat1, feat2):
    return (np.sum(feat1 * feat2) + 1) * 50

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image1', nargs='?', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test', '1.jpg'))
    parser.add_argument('image2', nargs='?', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test', '2.png'))
    parser.add_argument('--tuned', action='store_true', help='use the LFW fine-tuned head and its threshold')
    args = parser.parse_args()
    limit = tuned_threshold if args.tuned else threshold

    feats = []
    for fn in (args.image1, args.image2):
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit('cannot read image ' + fn)
        count, boxes, scores, landmarks, alignimgs, features = GetImageInfo(img, 5, tuned=args.tuned)
        if count == 0:
            raise SystemExit('no face found in ' + fn)
        feats.append(features[0])

    score = get_similarity(feats[0], feats[1])
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('score = ', score, '(threshold %.1f)' % limit)
    if score > limit:
        print('same person')
    else:
        print('different person')
