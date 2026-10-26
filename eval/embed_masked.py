# Cover nose and mouth on every LFW pair image and embed the covered version -> <lfw_dir>/masked.npz
# Usage: python eval/embed_masked.py <lfw_dir>   (run eval/embed_lfw.py first; ~20 min on CPU)
import os, sys, time, warnings
warnings.filterwarnings('ignore')
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo)
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from run import GetImageInfo

FILLS = ('surgical', 'black', 'white', 'skin', 'random', 'texture')

def cover_lower_face(img, lm, rng):
    """Paint over nose and mouth using 68-point landmarks; style and height vary per call."""
    lm = lm.reshape(68, 2)
    top = lm[rng.choice([28, 29, 30])]               # upper edge somewhere on the nose bridge
    jaw = lm[1:16]                                     # cheek to cheek around the chin
    poly = np.vstack([jaw, top[None]]).astype(np.int32)
    fill = FILLS[rng.integers(len(FILLS))]
    out = img.copy()
    if fill == 'texture':
        noise = rng.integers(0, 256, img.shape, dtype=np.uint8)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        out[mask > 0] = cv2.GaussianBlur(noise, (7, 7), 0)[mask > 0]
        return out
    if fill == 'skin':
        # colour sampled from the forehead/cheek area, like a hand over the face
        x, y = lm[27].astype(int)
        patch = img[max(y - 10, 0):y + 5, max(x - 10, 0):x + 10].reshape(-1, 3)
        colour = np.median(patch, 0) if len(patch) else np.array([150, 170, 200])
    else:
        colour = {'surgical': (225, 205, 170), 'black': (25, 25, 25), 'white': (240, 240, 240),
                  'random': tuple(int(v) for v in rng.integers(0, 256, 3))}[fill]
    cv2.fillPoly(out, [poly], tuple(int(c) for c in colour))
    return out

def centre_face(img, n, boxes):
    c = np.array(img.shape[1::-1]) / 2
    return int(np.argmin([np.linalg.norm((bx[:2] + bx[2:]) / 2 - c) for bx in boxes]))

if __name__ == '__main__':
    lfw = sys.argv[1]
    d = np.load(os.path.join(lfw, 'embeddings.npz'))
    paths = [str(p) for p in d['paths']]

    def embed(k):
        rng = np.random.default_rng(k)
        img = cv2.imread(os.path.join(lfw, 'lfw', paths[k]))
        n, boxes, _, lms, _, _ = GetImageInfo(img, 5)
        if n == 0:
            return None
        covered = cover_lower_face(img, lms[centre_face(img, n, boxes)], rng)
        n, boxes, _, _, _, feats = GetImageInfo(covered, 5)
        return feats[centre_face(covered, n, boxes)] if n else None

    t = time.perf_counter()
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(embed, range(len(paths))))
    found = np.array([f is not None for f in res])
    dim = d['F'].shape[1]
    Fm = np.stack([f if f is not None else np.zeros(dim, np.float32) for f in res])
    print('embedded %d covered images in %.0f s; face found in %.1f%%' % (len(paths), time.perf_counter() - t, found.mean() * 100))
    np.savez(os.path.join(lfw, 'masked.npz'), F=Fm, found=found)
    # save a few examples to eyeball the covering
    ex_dir = os.path.join(lfw, 'masked_examples'); os.makedirs(ex_dir, exist_ok=True)
    for k in range(0, 60, 10):
        img = cv2.imread(os.path.join(lfw, 'lfw', paths[k]))
        n, boxes, _, lms, _, _ = GetImageInfo(img, 5)
        if n:
            cv2.imwrite(os.path.join(ex_dir, '%d.jpg' % k), cover_lower_face(img, lms[centre_face(img, n, boxes)], np.random.default_rng(k)))
