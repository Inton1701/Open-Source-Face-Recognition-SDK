# Download LFW (if needed) and embed every image in pairs.txt -> <lfw_dir>/embeddings.npz
# Usage: python eval/embed_lfw.py <lfw_dir>      (~173 MB download, ~10 min on CPU)
import os, sys, tarfile, time, urllib.request, warnings
warnings.filterwarnings('ignore')
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo)
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from run import GetImageInfo

lfw = sys.argv[1]
os.makedirs(lfw, exist_ok=True)
for name, url in (('pairs.txt', 'https://ndownloader.figshare.com/files/5976006'),
                  ('lfw.tgz', 'https://ndownloader.figshare.com/files/5976018')):
    if not os.path.exists(os.path.join(lfw, name)):
        print('downloading', name)
        urllib.request.urlretrieve(url, os.path.join(lfw, name))
img_dir = os.path.join(lfw, 'lfw')
if not os.path.isdir(img_dir):
    with tarfile.open(os.path.join(lfw, 'lfw.tgz')) as t:
        t.extractall(lfw, filter='data')

lines = open(os.path.join(lfw, 'pairs.txt')).read().splitlines()
folds, per = map(int, lines[0].split())
pairs = []  # (fold, same, path1, path2)
for i, line in enumerate(lines[1:]):
    p = line.split('\t')
    fold = i // (2 * per)
    if len(p) == 3:
        pairs.append((fold, 1, '%s/%s_%04d.jpg' % (p[0], p[0], int(p[1])), '%s/%s_%04d.jpg' % (p[0], p[0], int(p[2]))))
    else:
        pairs.append((fold, 0, '%s/%s_%04d.jpg' % (p[0], p[0], int(p[1])), '%s/%s_%04d.jpg' % (p[2], p[2], int(p[3]))))
paths = sorted({p for _, _, a, b in pairs for p in (a, b)})
print(len(pairs), 'pairs,', len(paths), 'images')

def embed(path):
    img = cv2.imread(os.path.join(img_dir, path))
    n, boxes, scores, lms, aligns, feats = GetImageInfo(img, 5)
    if n == 0:
        return None
    # the LFW subject is the face nearest the image centre
    c = np.array(img.shape[1::-1]) / 2
    return feats[int(np.argmin([np.linalg.norm((bx[:2] + bx[2:]) / 2 - c) for bx in boxes]))]

t = time.perf_counter()
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(embed, paths))
print('embedded in %.0f s, no face found in %d images' % (time.perf_counter() - t, sum(f is None for f in res)))
dim = next(f for f in res if f is not None).shape[0]
F = np.stack([f if f is not None else np.zeros(dim, np.float32) for f in res])
found = np.array([f is not None for f in res])
idx = {p: i for i, p in enumerate(paths)}
P = np.array([(f, s, idx[a], idx[b]) for f, s, a, b in pairs])
np.savez(os.path.join(lfw, 'embeddings.npz'), F=F, found=found, P=P, paths=np.array(paths))
