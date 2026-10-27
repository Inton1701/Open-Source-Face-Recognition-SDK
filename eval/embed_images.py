# Embed the LFW pair images (clean or covered) with one model, with all networking blocked.
# Writes <lfw_dir>/emb_<model>_<set>.npz.  Models: fp (Faceplugin), s / l (InsightFace small / large),
# dlib (dlib ResNet + BlazeFace, face_dlib/).
# Usage: python eval/embed_images.py <lfw_dir> <fp|s|l|dlib> <clean|covered>
import os, socket, sys, time, warnings
warnings.filterwarnings('ignore')

def _blocked(*a, **k):
    raise OSError('network blocked: offline test')
socket.socket.connect = socket.socket.connect_ex = socket.getaddrinfo = socket.create_connection = _blocked

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo)
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

lfw, model, which = sys.argv[1:4]
paths = [str(p) for p in np.load(os.path.join(lfw, 'embeddings.npz'))['paths']]
if model == 'fp':
    from run import GetImageInfo
    def get(img):
        n, boxes, _, _, _, feats = GetImageInfo(img, 5)
        return n, boxes, feats
else:
    if model == 'dlib':
        from face_dlib.dlib_face import DlibFace
        m = DlibFace()
    else:
        from face_insight.insight import InsightFace
        m = InsightFace(model)
    def get(img):
        n, boxes, _, _, feats = m.get(img, 5)
        return n, boxes, feats

def embed(path):
    if which == 'clean':
        img = cv2.imread(os.path.join(lfw, 'lfw', path))
    else:
        img = cv2.imread(os.path.join(lfw, 'lfw_covered', path.replace('.jpg', '.png')))
    if img is None:
        return None
    n, boxes, feats = get(img)
    if n == 0:
        return None
    c = np.array(img.shape[1::-1]) / 2
    return feats[int(np.argmin([np.linalg.norm((np.asarray(b[:2]) + np.asarray(b[2:])) / 2 - c) for b in boxes]))]

t = time.perf_counter()
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(embed, paths))
dt = time.perf_counter() - t
dim = next(f for f in res if f is not None).shape[0]
F = np.stack([f if f is not None else np.zeros(dim, np.float32) for f in res]).astype(np.float32)
found = np.array([f is not None for f in res])
np.savez(os.path.join(lfw, 'emb_%s_%s.npz' % (model, which)), F=F, found=found, seconds=dt)
print('%s %s: %d images in %.0f s (%.0f ms each, 4 threads), face found %.1f%%, network blocked throughout'
      % (model, which, len(paths), dt, dt / len(paths) * 1000, found.mean() * 100), flush=True)
