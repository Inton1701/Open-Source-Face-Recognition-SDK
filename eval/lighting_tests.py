# Lighting and night tests: enrol the clean LFW photo, probe with the other photo of the pair under
# simulated lighting. Reports face found, genuine matched and impostors accepted at the shipped thresholds.
# Usage: python eval/lighting_tests.py <lfw_dir> [s] [l] [dlib] [--preview]
#   (needs eval/embed_lfw.py and eval/embed_images.py <lfw_dir> <s|l|dlib> clean first)
import os, sys, time, warnings
warnings.filterwarnings('ignore')
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo)
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

def gamma(img, g):
    return np.clip(255 * (img / 255.0) ** g, 0, 255).astype(np.uint8)

def sensor_noise(img, rng, sigma, shot=0.0):
    x = img.astype(np.float32)
    if shot:
        x = x + rng.normal(0, 1, x.shape) * np.sqrt(np.maximum(x, 0) * shot)
    return np.clip(x + rng.normal(0, sigma, x.shape), 0, 255).astype(np.uint8)

def dark(img, rng, level, sigma):
    x = (img.astype(np.float32) * level)
    return sensor_noise(x.astype(np.uint8), rng, sigma, shot=0.3)

def night_gain(img, rng):
    # very little light, camera auto-exposure amplifies it: bright but grainy and soft
    x = img.astype(np.float32) * 0.1
    x = x + rng.normal(0, 1, x.shape) * np.sqrt(np.maximum(x, 0) * 0.5) + rng.normal(0, 2.5, x.shape)
    x = cv2.GaussianBlur(np.clip(x * 7, 0, 255).astype(np.uint8), (3, 3), 0)
    return cv2.imdecode(cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, 60])[1], 1)

def infrared(img, rng):
    # night-vision camera: grey, high contrast, glowing skin, noisy
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = np.clip((g - 40) * 1.6, 0, 255)
    g = sensor_noise(cv2.GaussianBlur(g.astype(np.uint8), (3, 3), 0), rng, 10)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def side_light(img, rng):
    h, w = img.shape[:2]
    ramp = np.linspace(1.15, 0.12, w, dtype=np.float32)[None, :, None]
    if rng.random() < 0.5:
        ramp = ramp[:, ::-1]
    return np.clip(img * ramp, 0, 255).astype(np.uint8)

def backlit(img, rng):
    # bright window behind the person, face underexposed
    h, w = img.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    face = np.exp(-(((xx - w / 2) / (w * 0.22)) ** 2 + ((yy - h * 0.45) / (h * 0.3)) ** 2))[..., None]
    x = img.astype(np.float32) * (1.6 - 1.35 * face) + 70 * (1 - face)
    return np.clip(x, 0, 255).astype(np.uint8)

CONDITIONS = {
    'normal light (reference)': lambda im, r: im,
    'dim room (40%)': lambda im, r: dark(im, r, 0.4, 3),
    'dark night (15%)': lambda im, r: dark(im, r, 0.15, 4),
    'very dark night (6%)': lambda im, r: dark(im, r, 0.06, 4),
    'night, camera boosts brightness': night_gain,
    'infrared night camera': infrared,
    'side light (half in shadow)': side_light,
    'backlit (window behind)': backlit,
    'harsh sun / overexposed': lambda im, r: cv2.convertScaleAbs(im, alpha=1.9, beta=50),
    'warm bulb (orange cast)': lambda im, r: np.clip(gamma(im, 1.4) * np.array([0.55, 0.85, 1.15]), 0, 255).astype(np.uint8),
    'fluorescent (green cast)': lambda im, r: np.clip(im * np.array([0.95, 1.12, 0.8]), 0, 255).astype(np.uint8),
    'haze / low contrast': lambda im, r: cv2.addWeighted(im, 0.45, np.full_like(im, 170), 0.55, 0),
}
NIGHT = [c for c in CONDITIONS if 'night' in c or 'dim' in c or 'backlit' in c or 'side' in c]
from face_insight.insight import auto_enhance

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    lfw, models = args[0], args[1:] or ['s', 'l']
    d = np.load(os.path.join(lfw, 'embeddings.npz'))
    paths, (fold, same, a, b) = [str(p) for p in d['paths']], d['P'].T
    sel = np.where(fold < 2)[0]                       # 600 genuine + 600 impostor pairs
    load = lambda i: cv2.imread(os.path.join(lfw, 'lfw', paths[i]))

    if '--preview' in sys.argv:
        img = load(b[sel[0]])
        tiles = [cv2.putText(cv2.resize(fn(img, np.random.default_rng(0)), (200, 200)), str(k + 1), (6, 22), 0, 0.7, (0, 255, 255), 2)
                 for k, fn in enumerate(CONDITIONS.values())]
        cv2.imwrite(os.path.join(lfw, 'lighting_preview.jpg'), np.vstack([np.hstack(tiles[:6]), np.hstack(tiles[6:])]))
        print('\n'.join('%d. %s' % (k + 1, c) for k, c in enumerate(CONDITIONS)))
        sys.exit()

    from face_insight.insight import InsightFace, similarity, THRESHOLDS
    def load(m):
        if m == 'dlib':
            from face_dlib.dlib_face import DlibFace, THRESHOLDS as T
            return DlibFace(), T[m]
        return InsightFace(m), THRESHOLDS[m]
    for m in models:
        model, T = load(m)
        E = np.load(os.path.join(lfw, 'emb_%s_clean.npz' % m))
        enrolled, enrolled_ok = E['F'], E['found']

        def probe(args):
            k, cond, enhance = args
            img = CONDITIONS[cond](load(b[k]), np.random.default_rng(k))
            if enhance:
                img = auto_enhance(img)
            n, boxes, _, _, feats = model.get(img, 5)
            if n == 0 or not enrolled_ok[a[k]]:
                return None
            c = np.array(img.shape[1::-1]) / 2
            f = feats[int(np.argmin([np.linalg.norm((bx[:2] + bx[2:]) / 2 - c) for bx in boxes]))]
            return similarity(enrolled[a[k]], f)

        print('\n== %s (threshold %.1f), %d genuine + %d impostor pairs'
              % ({'s': 'InsightFace small', 'l': 'InsightFace large', 'dlib': 'dlib + BlazeFace'}[m], T, (same[sel] == 1).sum(), (same[sel] == 0).sum()))
        print('  %-34s %10s %15s %17s %13s' % ('condition', 'face found', 'genuine matched', 'impostors accepted', 'median score'))
        t0 = time.perf_counter()
        runs = [(c, False) for c in CONDITIONS] + [(c, True) for c in NIGHT if m == 's']
        for cond, enhance in runs:
            with ThreadPoolExecutor(4) as ex:
                s = list(ex.map(probe, [(k, cond, enhance) for k in sel]))
            found = np.array([x is not None for x in s])
            sc = np.array([x if x is not None else -1 for x in s])
            g, i = same[sel] == 1, same[sel] == 0
            print('  %-34s %9.1f%% %14.1f%% %16.2f%% %13.1f' % (cond + (' + auto-enhance' if enhance else ''), found.mean() * 100,
                  (sc[g] > T).mean() * 100, (sc[i] > T).mean() * 100, np.median(sc[g & found]) if (g & found).any() else float('nan')), flush=True)
        print('  (%.0f s)' % (time.perf_counter() - t0))
