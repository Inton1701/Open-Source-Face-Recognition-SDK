# InsightFace backend (SCRFD detector + ArcFace recogniser) run straight through onnxruntime, offline.
# Models: python face_insight/download_models.py  (one-time; copy face_insight/models/ to offline machines)
# Licence: InsightFace code is MIT, but its pretrained models are for non-commercial research use only.
import os
import cv2
import numpy as np
import onnxruntime as ort

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
PACKS = {'s': ('buffalo_s', 'det_500m.onnx', 'w600k_mbf.onnx'), 'l': ('buffalo_l', 'det_10g.onnx', 'w600k_r50.onnx')}
# same-person thresholds on the similarity() 0-100 scale: about 1% impostors accepted on LFW clean +
# nose/mouth-covered pairs (eval/compare_models.py)
THRESHOLDS = {'s': 60.3, 'l': 58.2}

# ArcFace 112x112 positions of: left eye, right eye, nose tip, left mouth corner, right mouth corner
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

def nms(boxes, scores, thresh=0.4):
    order, keep = scores.argsort()[::-1], []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    while order.size:
        i = order[0]; keep.append(i)
        w = np.maximum(0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]) + 1)
        h = np.maximum(0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]) + 1)
        iou = w * h / (areas[i] + areas[order[1:]] - w * h)
        order = order[1:][iou <= thresh]
    return keep

def gamma(channel, g):
    return np.clip(255 * (channel / 255.0) ** g, 0, 255).astype(np.uint8)

def auto_enhance(img):
    """Brighten dark photos (night, backlight): gamma by mean lightness, then CLAHE. Bright photos pass through."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    mean = lab[..., 0].mean()
    if mean >= 90:
        return img
    lab[..., 0] = gamma(lab[..., 0], max(0.35, np.log(0.45) / np.log(max(mean, 1) / 255)))
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

class InsightFace:
    def __init__(self, size='s', model_dir=MODEL_DIR, det_size=640, det_thresh=0.5, fallback_thresh=None,
                 flip=False, enhance=False):
        """flip: average the face with its mirror image; enhance: brighten dark photos before detection;
        fallback_thresh: when nothing passes det_thresh, retry at this score and keep the best face."""
        pack, det, rec = PACKS[size]
        opts = ort.SessionOptions()
        # idle threads spinning between runs starve the other session: large model 615 -> 135 ms per photo
        opts.add_session_config_entry('session.intra_op.allow_spinning', '0')
        self.det = ort.InferenceSession(os.path.join(model_dir, pack, det), opts, providers=['CPUExecutionProvider'])
        self.rec = ort.InferenceSession(os.path.join(model_dir, pack, rec), opts, providers=['CPUExecutionProvider'])
        self.det_size, self.det_thresh, self.fallback_thresh = det_size, det_thresh, fallback_thresh
        self.flip, self.enhance = flip, enhance
        self.det_in, self.rec_in = self.det.get_inputs()[0].name, self.rec.get_inputs()[0].name
        self._centers = {}

    def _anchors(self, h, w, stride):
        key = (h, w, stride)
        if key not in self._centers:
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32).reshape(-1, 2) * stride
            self._centers[key] = np.repeat(c, 2, axis=0)  # 2 anchors per location
        return self._centers[key]

    def detect(self, img, thresh=None):
        """BGR image -> boxes (n,4), scores (n,), 5-point landmarks (n,5,2), in image pixels, best first."""
        thresh = self.det_thresh if thresh is None else thresh
        s = self.det_size
        scale = s / max(img.shape[:2])
        resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        canvas = np.zeros((s, s, 3), np.uint8); canvas[:resized.shape[0], :resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128, (s, s), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.det.run(None, {self.det_in: blob})
        boxes, scores, kpss = [], [], []
        for i, stride in enumerate((8, 16, 32)):
            sc, bb, kp = outs[i][:, 0], outs[i + 3] * stride, outs[i + 6] * stride
            ctr = self._anchors(s // stride, s // stride, stride)
            keep = sc >= thresh
            c = ctr[keep]
            boxes.append(np.hstack([c - bb[keep][:, :2], c + bb[keep][:, 2:]]))
            scores.append(sc[keep])
            kpss.append(c[:, None, :] + kp[keep].reshape(-1, 5, 2))
        boxes, scores, kpss = np.vstack(boxes) / scale, np.concatenate(scores), np.vstack(kpss) / scale
        keep = nms(boxes, scores) if len(scores) else []
        return boxes[keep], scores[keep], kpss[keep]

    def embed(self, img, kps):
        """Align one face by its 5 landmarks and return its L2-normalised 512-d embedding."""
        M = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARCFACE_DST, method=cv2.LMEDS)[0]
        face = cv2.warpAffine(img, M, (112, 112), borderValue=0)
        blob = cv2.dnn.blobFromImage(face, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        f = self.rec.run(None, {self.rec_in: blob})[0][0]
        f = f / np.linalg.norm(f)
        if self.flip:  # separate run: the models declare batch 1 and a batch of 2 is ~10x slower on the large one
            m = self.rec.run(None, {self.rec_in: np.ascontiguousarray(blob[..., ::-1])})[0][0]
            f = f + m / np.linalg.norm(m)
        return f / np.linalg.norm(f), face

    def get(self, img, max_faces=10):
        """Same idea as run.GetImageInfo: (count, boxes, scores, aligned faces, embeddings)."""
        if self.enhance:
            img = auto_enhance(img)
        boxes, scores, kpss = self.detect(img)
        if len(boxes) == 0 and self.fallback_thresh:
            boxes, scores, kpss = (x[:1] for x in self.detect(img, self.fallback_thresh))
        boxes, scores, kpss = boxes[:max_faces], scores[:max_faces], kpss[:max_faces]
        out = [self.embed(img, k) for k in kpss]
        return len(boxes), list(boxes), list(scores), [o[1] for o in out], [o[0] for o in out]

def similarity(f1, f2):
    """Cosine similarity mapped to 0-100 like run.get_similarity."""
    return (float(np.dot(f1, f2)) + 1) * 50
