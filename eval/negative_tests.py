# Negative tests: bad inputs and non-faces (asserted), then degraded genuine/impostor LFW pairs (reported).
# Usage: python eval/negative_tests.py [<lfw_dir>] [--tuned]   (LFW part needs eval/embed_lfw.py first)
import os, sys, time, warnings
warnings.filterwarnings('ignore')
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo)
import cv2
import numpy as np
from run import GetImageInfo, get_similarity, threshold, tuned_threshold

args = [a for a in sys.argv[1:] if a != '--tuned']
tuned = '--tuned' in sys.argv
T = tuned_threshold if tuned else threshold
rng = np.random.default_rng(0)
face = cv2.imread(os.path.join(repo, 'test', '1.jpg'))
failures = []

def check(name, img, expect):
    """expect: an exception type, or a number of faces."""
    t = time.perf_counter()
    try:
        got = GetImageInfo(img, 5)[0]
        out = '%d face(s)' % got
    except Exception as e:
        got, out = type(e), '%s: %s' % (type(e).__name__, e)
    passed = got == expect
    if not passed:
        failures.append(name)
    print('  %s %-30s -> %s  (%.0f ms)' % ('ok  ' if passed else 'FAIL', name, out, (time.perf_counter() - t) * 1000))

print('== Bad / unusual inputs')
check('missing file (imread -> None)', cv2.imread('does-not-exist.jpg'), ValueError)
check('not an array', 'test/1.jpg', TypeError)
check('empty array', np.zeros((0, 0, 3), np.uint8), ValueError)
check('float32 image 0-1', face.astype(np.float32) / 255, ValueError)
check('5-channel image', np.zeros((64, 64, 5), np.uint8), ValueError)
check('1x1 pixel', np.zeros((1, 1, 3), np.uint8), 0)
check('grayscale (2-D) image', cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), 1)
check('BGRA 4-channel image', cv2.cvtColor(face, cv2.COLOR_BGR2BGRA), 1)
check('non-contiguous (sliced) image', face[:, ::2], 1)
check('huge 6000x4000 image', cv2.resize(face, (6000, 4000)), 1)

print('== Non-face images (expect 0 faces)')
check('blank grey', np.full((480, 640, 3), 127, np.uint8), 0)
check('black', np.zeros((480, 640, 3), np.uint8), 0)
check('random noise', rng.integers(0, 256, (480, 640, 3), dtype=np.uint8), 0)
grad = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
check('gradient', cv2.merge([grad] * 3), 0)
board = (np.indices((480, 640)).sum(0) // 40 % 2 * 255).astype(np.uint8)
check('checkerboard', cv2.merge([board] * 3), 0)
smiley = np.full((480, 640, 3), 200, np.uint8)
cv2.circle(smiley, (320, 240), 120, (60, 90, 160), -1)
cv2.circle(smiley, (280, 210), 15, (0, 0, 0), -1); cv2.circle(smiley, (360, 210), 15, (0, 0, 0), -1)
cv2.ellipse(smiley, (320, 300), (50, 20), 0, 0, 180, (0, 0, 0), 5)
check('smiley drawing (known false detection)', smiley, 1)

if args:
    lfw = args[0]
    d = np.load(os.path.join(lfw, 'embeddings.npz'))
    F, found, P, paths = d['F'], d['found'], d['P'], d['paths']
    gen = [r for r in P if r[1] == 1 and found[r[2]] and found[r[3]]][:150]
    imp = [r for r in P if r[1] == 0 and found[r[2]] and found[r[3]]][:150]
    load = lambda i: cv2.imread(os.path.join(lfw, 'lfw', str(paths[i])))

    def feat(img):
        n, boxes, _, _, _, fs = GetImageInfo(img, 5, tuned=tuned)
        if n == 0:
            return None
        c = np.array(img.shape[1::-1]) / 2
        return fs[int(np.argmin([np.linalg.norm((bx[:2] + bx[2:]) / 2 - c) for bx in boxes]))]

    def rot(img, deg):
        h, w = img.shape[:2]
        return cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1), (w, h))

    def cover(img, top, bottom):
        img = img.copy(); h = img.shape[0]
        img[int(h * top):int(h * bottom)] = 0
        return img

    degrade = {
        'none (reference)': lambda im: im,
        'blur 9px': lambda im: cv2.GaussianBlur(im, (9, 9), 0),
        'blur 21px': lambda im: cv2.GaussianBlur(im, (21, 21), 0),
        'dark (25% brightness)': lambda im: (im * .25).astype(np.uint8),
        'overexposed': lambda im: cv2.convertScaleAbs(im, alpha=2.2, beta=60),
        'low-res 40px wide': lambda im: cv2.resize(cv2.resize(im, (40, 40)), im.shape[1::-1]),
        'jpeg quality 5': lambda im: cv2.imdecode(cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 5])[1], 1),
        'noise sigma 25': lambda im: np.clip(im + rng.normal(0, 25, im.shape), 0, 255).astype(np.uint8),
        'lower face covered (mask)': lambda im: cover(im, .55, 1),
        'eyes covered (sunglasses)': lambda im: cover(im, .35, .5),
        'rotated 20 deg': lambda im: rot(im, 20),
        'rotated 45 deg': lambda im: rot(im, 45),
        'upside down': lambda im: cv2.flip(im, -1),
    }
    # reference features with the same head as the degraded ones
    ref = {i: feat(load(i)) for r in gen + imp for i in (r[2],)}

    print('\n== Degraded genuine faces (%d same-person LFW pairs, threshold %.1f)' % (len(gen), T))
    print('  %-28s%11s%15s%14s' % ('degradation', 'face found', 'still matched', 'median score'))
    for name, fn in degrade.items():
        scores = [get_similarity(ref[r[2]], f) for r in gen for f in [feat(fn(load(r[3])))] if f is not None and ref[r[2]] is not None]
        print('  %-28s%10.0f%%%14.0f%%%14.1f' % (name, len(scores) / len(gen) * 100,
              sum(s > T for s in scores) / len(gen) * 100, np.median(scores) if scores else float('nan')))

    print('\n== Degraded impostors (%d different-person pairs, false accepts should stay near 0)' % len(imp))
    for name in ('none (reference)', 'blur 21px', 'dark (25% brightness)', 'lower face covered (mask)', 'low-res 40px wide'):
        scores = [get_similarity(ref[r[2]], f) for r in imp for f in [feat(degrade[name](load(r[3])))] if f is not None and ref[r[2]] is not None]
        print('  %-28s false accepts %d/%d' % (name, sum(s > T for s in scores), len(scores)))

if failures:
    raise SystemExit('FAILED: ' + ', '.join(failures))
print('\nall input checks passed')
