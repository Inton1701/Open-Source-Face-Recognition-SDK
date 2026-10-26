# LFW View-2 accuracy: SDK as shipped, plus the linear head cross-validated on unseen people.
# Usage: python eval/eval_lfw.py <lfw_dir>   (run eval/embed_lfw.py first)
import os, sys
import numpy as np
from finetune_head import train_head, project, best_threshold

d = np.load(os.path.join(sys.argv[1], 'embeddings.npz'))
F, found, P = d['F'], d['found'], d['P']
fold, same, a, b = P.T
ok = found[a] & found[b]

def scores_of(X):
    # SDK score; a pair with a missed face counts as "different person"
    return np.where(ok, (np.sum(X[a] * X[b], 1) + 1) * 50, 0.0)

def acc(s, y, t):
    return float(((s > t) == y).mean())

def report(name, s, t):
    print('\n== %s (threshold %.1f)' % (name, t))
    pos, neg = s[same == 1], s[same == 0]
    print('  accuracy %.2f%% | true matches accepted %.1f%% | impostors accepted %.2f%% (%d/%d)'
          % (acc(s, same, t) * 100, (pos > t).mean() * 100, (neg > t).mean() * 100, (neg > t).sum(), len(neg)))
    neg_sorted = np.sort(neg)[::-1]
    for far in (1e-2, 1e-3):
        tf = neg_sorted[max(int(far * len(neg)) - 1, 0)]
        print('  at %.1f%% impostor rate: threshold %.1f, true matches accepted %.1f%%' % (far * 100, tf, (pos > tf).mean() * 100))

print('pairs %d, pairs with a missed face %d' % (len(P), (~ok).sum()))
base = scores_of(F)
report('Baseline (SDK as shipped)', base, 75)
cv = []
for k in range(10):
    tr, te = fold != k, fold == k
    cv.append(acc(base[te], same[te], best_threshold(base[tr], same[tr])))
print('  10-fold calibrated threshold: accuracy %.2f%% +/- %.2f' % (np.mean(cv) * 100, np.std(cv) * 100))

# Head: train on 9 folds, test on the 10th (LFW folds hold different people); shift so 75 = fold threshold
tuned = np.zeros(len(P))
for k in range(10):
    tr, te = (fold != k) & ok, fold == k
    s = scores_of(project(F, train_head(F, a[tr], b[tr], same[tr])))
    tuned[te] = s[te] - best_threshold(s[fold != k], same[fold != k]) + 75
report('Fine-tuned head, 10-fold cross-validated', tuned, 75)
