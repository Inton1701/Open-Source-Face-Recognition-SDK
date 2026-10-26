# Covered-face (nose + mouth) accuracy on LFW: raw features vs clean-trained head vs mask-aware head,
# 10-fold cross-validated on unseen people. --save trains the mask-aware head on all folds and
# writes it to face_feature/tuned_head.npz.
# Usage: python eval/eval_masked.py <lfw_dir> [--save]   (needs embed_lfw.py and embed_masked.py first)
import os, sys
import numpy as np
from finetune_head import train_head, project, best_threshold

lfw = sys.argv[1]
d, m = np.load(os.path.join(lfw, 'embeddings.npz')), np.load(os.path.join(lfw, 'masked.npz'))
N = len(d['F'])
F = np.vstack([d['F'], m['F']])                     # index k = clean image k, k + N = covered image k
found = np.concatenate([d['found'], m['found'] & d['found']])
fold, same, a, b = d['P'].T
CONDS = {'clean vs clean': (a, b), 'clean vs covered': (a, b + N), 'covered vs covered': (a + N, b + N)}

def scores(X, ia, ib):
    ok = found[ia] & found[ib]
    return np.where(ok, (np.sum(X[ia] * X[ib], 1) + 1) * 50, 0.0)  # missed face = "different"

def mixed(sel, conds=CONDS):
    """Stack pair indices of the selected pairs across conditions."""
    ia = np.concatenate([conds[c][0][sel] for c in conds]); ib = np.concatenate([conds[c][1][sel] for c in conds])
    return ia, ib, np.tile(same[sel], len(conds))

def train_on(sel, conds):
    ia, ib, y = mixed(sel, conds)
    ok = found[ia] & found[ib]
    # covered vs clean in both directions, so the head does not care which side is covered
    if 'clean vs covered' in conds:
        ia, ib, y, ok = np.r_[ia, ib], np.r_[ib, ia], np.r_[y, y], np.r_[ok, ok]
    return train_head(F, ia[ok], ib[ok], y[ok])

methods = {}
# 1. raw features, default threshold 75
methods['As shipped (threshold 75)'] = {c: scores(F, *CONDS[c]) - 75 for c in CONDS}
# 2. raw features, one threshold tuned on mixed training folds
# 3. head trained on clean pairs only (current tuned_head)
# 4. head trained on clean + covered pairs
def capped_threshold(s, y, rate=0.01):
    """Lowest threshold that keeps the impostor acceptance rate at or below `rate`."""
    neg = np.sort(s[y == 0])[::-1]
    return float(neg[int(rate * len(neg))])

out2, out3, out4, out5 = ({c: np.zeros(len(same)) for c in CONDS} for _ in range(4))
clean_only = {'clean vs clean': CONDS['clean vs clean']}
for k in range(10):
    tr, te = fold != k, fold == k
    ia, ib, y = mixed(tr)
    t = best_threshold(scores(F, ia, ib), y)
    for c in CONDS: out2[c][te] = scores(F, *CONDS[c])[te] - t
    Z = project(F, train_on(tr, clean_only))
    t = best_threshold(scores(Z, *CONDS['clean vs clean'])[tr], same[tr])
    for c in CONDS: out3[c][te] = scores(Z, *CONDS[c])[te] - t
    Z = project(F, train_on(tr, CONDS))
    s = scores(Z, ia, ib)
    t, ts = best_threshold(s, y), capped_threshold(s, y)
    for c in CONDS:
        out4[c][te] = scores(Z, *CONDS[c])[te] - t
        out5[c][te] = scores(Z, *CONDS[c])[te] - ts
    print('fold %d done' % k, flush=True)
methods['As shipped, threshold re-tuned'] = out2
methods['Current tuned head (clean training)'] = out3
methods['Mask-aware head, balanced threshold'] = out4
methods['Mask-aware head, strict threshold (<=1% impostors in training)'] = out5

print('\nface found in covered images: %.1f%%' % (m['found'][d['found']].mean() * 100))
for name, res in methods.items():
    print('\n== %s' % name)
    for c, s in res.items():
        pos, neg = s[same == 1], s[same == 0]
        print('  %-20s accuracy %6.2f%% | genuine matched %5.1f%% | impostors accepted %5.2f%%'
              % (c, ((s > 0) == same).mean() * 100, (pos > 0).mean() * 100, (neg > 0).mean() * 100))

if '--save' in sys.argv:
    W = train_on(np.ones(len(same), bool), CONDS)
    ia, ib, y = mixed(np.ones(len(same), bool))
    s = scores(project(F, W), ia, ib)
    t, ts = best_threshold(s, y), capped_threshold(s, y)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'face_feature', 'tuned_head.npz')
    np.savez(path, W=W.astype(np.float32), threshold=np.float32(ts), threshold_balanced=np.float32(t))
    print('\nsaved mask-aware head to %s (strict threshold %.2f, balanced %.2f)' % (path, ts, t))
