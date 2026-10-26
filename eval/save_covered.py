# Write the nose-and-mouth covered version of every LFW pair image to <lfw_dir>/lfw_covered/,
# so every model is tested on exactly the same pictures.
# Usage: python eval/save_covered.py <lfw_dir>   (run eval/embed_lfw.py first)
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from face_detect.detect_imgs import get_face_boundingbox
from face_landmark.GetLandmark import get_face_landmark
from embed_masked import cover_lower_face

lfw = sys.argv[1]
paths = [str(p) for p in np.load(os.path.join(lfw, 'embeddings.npz'))['paths']]

def save(k):
    img = cv2.imread(os.path.join(lfw, 'lfw', paths[k]))
    boxes, _ = get_face_boundingbox(img)
    if len(boxes) == 0:
        return False
    c = np.array(img.shape[1::-1]) / 2
    box = boxes[int(np.argmin([np.linalg.norm((bx[:2] + bx[2:]).numpy() / 2 - c) for bx in boxes]))]
    lm = get_face_landmark(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), box).data.numpy()
    out = os.path.join(lfw, 'lfw_covered', paths[k].replace('.jpg', '.png'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, cover_lower_face(img, lm, np.random.default_rng(k)))
    return True

with ThreadPoolExecutor(4) as ex:
    ok = list(ex.map(save, range(len(paths))))
print('saved %d of %d covered images' % (sum(ok), len(paths)))
