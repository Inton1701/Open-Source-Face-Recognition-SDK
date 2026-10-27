# Measure InsightFace improvements one by one: detector fallback, mirror averaging, auto-enhance of dark photos.
# Thresholds are set on LFW folds 2-4 (clean + covered, ~1% impostors) and everything is scored on folds 0-1
# (different people): clean, covered and 11 lighting conditions. Networking is blocked.
# Usage: python eval/improve_insight.py <lfw_dir> <s|l>   (needs embed_lfw.py and save_covered.py first)
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
from face_insight.insight import InsightFace, auto_enhance
from lighting_tests import CONDITIONS

lfw, size = sys.argv[1], sys.argv[2]
d = np.load(os.path.join(lfw, 'embeddings.npz'))
paths, (fold, same, a, b) = [str(p) for p in d['paths']], d['P'].T
model = InsightFace(size)
VARIANTS = {  # name: (feature key, found key)
    'as is': ('Fp', 'ok'), '+ detector fallback': ('Fp', 'ok_fb'), '+ mirror': ('Ff', 'ok'),
    '+ auto-enhance': ('Ep', 'eok'), 'all three': ('Ef', 'eok_fb')}

def analyse(img):
    """Detect once at 0.3: faces >= 0.5 are the plain result, else the best >= 0.3 is the fallback."""
    boxes, scores, kpss = model.detect(img, 0.3)
    if len(boxes) == 0:
        return None
    strong = scores >= model.det_thresh
    if strong.any():
        boxes, kpss = boxes[strong], kpss[strong]
    else:
        boxes, kpss = boxes[:1], kpss[:1]
    c = np.array(img.shape[1::-1]) / 2
    k = kpss[int(np.argmin([np.linalg.norm((bx[:2] + bx[2:]) / 2 - c) for bx in boxes]))]
    fp, face = model.embed(img, k)
    blob = cv2.dnn.blobFromImage(np.ascontiguousarray(face[:, ::-1]), 1 / 127.5, (112, 112), (127.5,) * 3, swapRB=True)
    fl = model.rec.run(None, {model.rec_in: blob})[0][0]
    ff = fp + fl / np.linalg.norm(fl)  # same as InsightFace(flip=True)
    return fp, ff / np.linalg.norm(ff), bool(strong.any())

def one(img):
    r = analyse(img)
    out = {'Fp': r[0] if r else None, 'Ff': r[1] if r else None, 'ok': bool(r and r[2]), 'ok_fb': r is not None}
    enh = auto_enhance(img)
    if enh is not img:
        r = analyse(enh)
    out.update(Ep=r[0] if r else None, Ef=r[1] if r else None, eok=bool(r and r[2]), eok_fb=r is not None)
    return out

def run(items, label):
    cache = os.path.join(lfw, 'improve_%s_%s.npz' % (size, label))
    if os.path.exists(cache):
        z = np.load(cache); return {k: z[k] for k in z.files}
    t = time.perf_counter()
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda f: one(f()), items))
    out = {}
    for k in ('Fp', 'Ff', 'Ep', 'Ef'):
        out[k] = np.stack([r[k] if r[k] is not None else np.zeros(512, np.float32) for r in res])
    for k in ('ok', 'ok_fb', 'eok', 'eok_fb'):
        out[k] = np.array([r[k] for r in res])
    np.savez(cache, **out)
    print('  embedded %-34s %5d images, %.0f ms each' % (label, len(items), (time.perf_counter() - t) / len(items) * 1000), flush=True)
    return out

pairs = np.where(fold < 5)[0]
imgs = sorted(set(a[pairs]) | set(b[pairs]))
pos = {i: n for n, i in enumerate(imgs)}
A, B = np.array([pos[i] for i in a[pairs]]), np.array([pos[i] for i in b[pairs]])
Y, test = same[pairs], fold[pairs] < 2
clean = run([lambda i=i: cv2.imread(os.path.join(lfw, 'lfw', paths[i])) for i in imgs], 'clean')
covered = run([lambda i=i: cv2.imread(os.path.join(lfw, 'lfw_covered', paths[i].replace('.jpg', '.png'))) for i in imgs], 'covered')

def scores(X, i, Z, j, var):
    f, ok = VARIANTS[var]
    return np.where(X[ok][i] & Z[ok][j], np.sum(X[f][i] * Z[f][j], 1), -np.inf)

def strict(s, y, rate=0.01):
    neg = np.sort(s[y == 0])[::-1]
    return neg[int(rate * len(neg))]

conds = {'clean vs clean': (clean, clean), 'clear vs covered': (clean, covered), 'covered vs covered': (covered, covered)}
TH = {}
print('\n== InsightFace %s: thresholds from folds 2-4, scored on folds 0-1 (%d genuine + %d impostor pairs)'
      % ({'s': 'small', 'l': 'large'}[size], (Y[test] == 1).sum(), (Y[test] == 0).sum()))
for var in VARIANTS:
    s = {c: scores(x, A, z, B, var) for c, (x, z) in conds.items()}
    TH[var] = strict(np.concatenate([s[c][~test] for c in s]), np.tile(Y[~test], len(s)))
    line = ' | '.join('%s %5.1f%% (imp %.2f%%)' % (c, (s[c][test & (Y == 1)] > TH[var]).mean() * 100, (s[c][test & (Y == 0)] > TH[var]).mean() * 100) for c in s)
    print('  %-20s threshold %5.1f | %s' % (var, (TH[var] + 1) * 50, line))

print('\n  Lighting (enrolled clear photo in normal light, probe under the condition): genuine matched / impostors accepted')
print('  %-34s' % 'condition' + ''.join('%22s' % v for v in VARIANTS))
tp = np.where(test)[0]
for cond, fn in CONDITIONS.items():
    if cond.startswith('normal'):
        continue
    probe = run([lambda i=i: fn(cv2.imread(os.path.join(lfw, 'lfw', paths[i])), np.random.default_rng(i)) for i in b[pairs][tp]], cond.split(' (')[0].replace(' ', '_').replace(',', '').replace('/', ''))
    row = []
    for var in VARIANTS:
        f, ok = VARIANTS[var]
        s = np.where(clean[ok][A[tp]] & probe[ok], np.sum(clean[f][A[tp]] * probe[f], 1), -np.inf)
        row.append('%5.1f%% / %4.2f%%' % ((s[Y[tp] == 1] > TH[var]).mean() * 100, (s[Y[tp] == 0] > TH[var]).mean() * 100))
    print('  %-34s' % cond + ''.join('%22s' % r for r in row), flush=True)
print('\n  thresholds (0-100 scale): ' + ', '.join('%s %.1f' % (v, (t + 1) * 50) for v, t in TH.items()))
