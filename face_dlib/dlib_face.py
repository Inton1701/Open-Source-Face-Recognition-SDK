# Free-licence backend: MediaPipe BlazeFace full-range detector (Apache-2.0, run with LiteRT) + dlib 5-point
# landmarks and ResNet recogniser (public domain / CC0). Same interface as face_insight.insight.InsightFace.
# Models: python face_dlib/download_models.py  (one-time; copy face_dlib/models/ to offline machines)
# Licence notes: dlib's recogniser was trained partly on VGG Face and FaceScrub (research-only datasets); CC0 does
# not clear rights in training data. BlazeFace decoding below follows MediaPipe's face_detection graph
# (Copyright The MediaPipe Authors, Apache-2.0): 192x192 letterbox in [-1, 1], 48x48 anchors, weighted NMS.
import os, threading
import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
# same-person threshold on the similarity() 0-100 scale, set like face_insight's (eval/compare_models.py)
THRESHOLDS = {'dlib': None}

def iou(box, boxes):
    w = np.maximum(0, np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]))
    h = np.maximum(0, np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]))
    inter = w * h
    return inter / ((box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter)

def weighted_nms(boxes, scores, kps, thresh=0.3):
    """MediaPipe's WEIGHTED suppression: overlapping detections are averaged by score instead of dropped."""
    order, out = scores.argsort()[::-1], []
    while order.size:
        overlap = iou(boxes[order[0]], boxes[order]) > thresh
        group, w = order[overlap], scores[order[overlap]][:, None]
        out.append(((boxes[group] * w).sum(0) / w.sum(), scores[order[0]], (kps[group] * w[:, :, None]).sum(0) / w.sum()))
        order = order[~overlap]
    return out

class BlazeFace:
    def __init__(self, path, threads=1):
        from ai_edge_litert.interpreter import Interpreter
        self.net = Interpreter(model_path=path, num_threads=threads)
        self.net.allocate_tensors()
        inp = self.net.get_input_details()[0]
        self.inp, self.size = inp['index'], int(inp['shape'][1])
        outs = self.net.get_output_details()
        self.box_out = next(o['index'] for o in outs if o['shape'][-1] == 16)
        self.score_out = next(o['index'] for o in outs if o['shape'][-1] == 1)
        n = self.size // 4  # full-range model: one anchor per cell at stride 4
        ys, xs = np.mgrid[:n, :n]
        self.anchors = np.stack([(xs.ravel() + 0.5) / n, (ys.ravel() + 0.5) / n], 1).astype(np.float32)

    def detect(self, img, thresh=0.5):
        """BGR image -> boxes (n,4), scores (n,), 6 keypoints (n,6,2) in image pixels, best first.
        Keypoints: eye on the image left, eye on the image right, nose tip, mouth centre, two ear tragions."""
        h, w = img.shape[:2]
        s = max(h, w)
        y0, x0 = (s - h) // 2, (s - w) // 2
        canvas = np.zeros((s, s, 3), np.uint8); canvas[y0:y0 + h, x0:x0 + w] = img
        x = cv2.cvtColor(cv2.resize(canvas, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        self.net.set_tensor(self.inp, (x.astype(np.float32) / 127.5 - 1)[None])
        self.net.invoke()
        raw = self.net.get_tensor(self.box_out).reshape(-1, 16) / self.size
        scores = 1 / (1 + np.exp(-np.clip(self.net.get_tensor(self.score_out).reshape(-1), -100, 100)))
        keep = scores >= thresh
        if not keep.any():
            return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 6, 2))
        a, r = self.anchors[keep], raw[keep]
        c = r[:, :2] + a
        boxes = np.hstack([c - r[:, 2:4] / 2, c + r[:, 2:4] / 2])
        kps = r[:, 4:].reshape(-1, 6, 2) + a[:, None]
        dets = weighted_nms(boxes, scores[keep], kps)
        off = np.array([x0, y0], np.float32)
        boxes = np.array([d[0] for d in dets]) * s - np.tile(off, 2)
        return boxes, np.array([d[1] for d in dets]), np.array([d[2] for d in dets]) * s - off

class DlibFace:
    def __init__(self, model_dir=MODEL_DIR, det_thresh=0.5):
        files = [os.path.join(model_dir, f) for f in ('face_detection_full_range.tflite',
                 'shape_predictor_5_face_landmarks.dat', 'dlib_face_recognition_resnet_model_v1.dat')]
        if not all(map(os.path.exists, files)):
            raise FileNotFoundError('face_dlib models are missing; run: python face_dlib/download_models.py')
        self.files, self.det_thresh = files, det_thresh
        self._local = threading.local()  # LiteRT interpreters and dlib nets are not thread-safe
        self._parts()

    def _parts(self):
        if not hasattr(self._local, 'det'):
            import dlib
            self._local.det = BlazeFace(self.files[0])
            self._local.sp, self._local.rec = dlib.shape_predictor(self.files[1]), dlib.face_recognition_model_v1(self.files[2])
        return self._local

    def detect(self, img, thresh=None):
        return self._parts().det.detect(img, self.det_thresh if thresh is None else thresh)

    def embed(self, rgb, box):
        """dlib 5 landmarks inside the detector box -> 150x150 dlib chip -> L2-normalised 128-d embedding."""
        import dlib
        p = self._parts()
        shape = p.sp(rgb, dlib.rectangle(*(int(round(v)) for v in box)))
        chip = dlib.get_face_chip(rgb, shape, 150, 0.25)
        f = np.array(p.rec.compute_face_descriptor(chip), np.float32)
        return f / np.linalg.norm(f), cv2.cvtColor(chip, cv2.COLOR_RGB2BGR)

    def get(self, img, max_faces=10):
        """Same as InsightFace.get: (count, boxes, scores, aligned faces, embeddings)."""
        boxes, scores, _ = self.detect(img)
        boxes, scores = boxes[:max_faces], scores[:max_faces]
        rgb = np.ascontiguousarray(img[..., ::-1])
        out = [self.embed(rgb, b) for b in boxes]
        return len(boxes), list(boxes), list(scores), [o[1] for o in out], [o[0] for o in out]

def similarity(f1, f2):
    """Cosine similarity mapped to 0-100 like face_insight.insight.similarity."""
    return (float(np.dot(f1, f2)) + 1) * 50
