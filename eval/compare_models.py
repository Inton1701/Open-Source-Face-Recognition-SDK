# Compare models on LFW clean and nose+mouth covered pairs, 10-fold cross-validated.
# One threshold per model, picked on the training folds (pooled over all conditions) so that
# at most 1% of impostors get in; reported on the held-out fold.
# Usage: python eval/compare_models.py <lfw_dir> [fp s l]   (needs eval/embed_images.py outputs)
import os, sys
import numpy as np

lfw = sys.argv[1]
models = sys.argv[2:] or ['fp', 's', 'l']
NAMES = {'fp': 'Faceplugin (current)', 's': 'InsightFace small (MobileFaceNet)', 'l': 'InsightFace large (ResNet50)'}
fold, same, a, b = np.load(os.path.join(lfw, 'embeddings.npz'))['P'].T

def load(m, which):
    d = np.load(os.path.join(lfw, 'emb_%s_%s.npz' % (m, which)))
    return d['F'], d['found']

def sim(F1, ok1, i, F2, ok2, j):
    return np.where(ok1[i] & ok2[j], np.sum(F1[i] * F2[j], 1), -np.inf)  # missed face = no match

def strict_threshold(s, y, rate=0.01):
    neg = np.sort(s[y == 0])[::-1]
    return neg[int(rate * len(neg))]

for m in models:
    (C, okc), (V, okv) = load(m, 'clean'), load(m, 'covered')
    conds = {
        'clean vs clean': sim(C, okc, a, C, okc, b),
        'clear vs covered': sim(C, okc, a, V, okv, b),
        'covered vs covered': sim(V, okv, a, V, okv, b),
    }
    # enrolled with a clear and a covered photo, probe covered: best of the two
    conds['clear+covered enrolled vs covered'] = np.maximum(conds['clear vs covered'], conds['covered vs covered'])
    pooled = ('clean vs clean', 'clear vs covered', 'covered vs covered')
    hit = {c: np.zeros(len(same), bool) for c in conds}
    acc_clean = []
    for k in range(10):
        tr, te = fold != k, fold == k
        t = strict_threshold(np.concatenate([conds[c][tr] for c in pooled]), np.tile(same[tr], len(pooled)))
        for c in conds:
            hit[c][te] = conds[c][te] > t
        # standard LFW accuracy: best threshold on clean training folds
        s = conds['clean vs clean']
        ts = np.unique(s[tr][np.isfinite(s[tr])])
        tb = ts[np.argmax([((s[tr] > x) == same[tr]).mean() for x in ts])]
        acc_clean.append(((s[te] > tb) == same[te]).mean())
    print('\n== %s' % NAMES[m])
    print('  LFW accuracy (clean, standard protocol): %.2f%% +/- %.2f | face found: clean %.1f%%, covered %.1f%%'
          % (np.mean(acc_clean) * 100, np.std(acc_clean) * 100, okc.mean() * 100, okv.mean() * 100))
    for c in conds:
        print('  %-36s genuine matched %5.1f%% | impostors accepted %5.2f%%'
              % (c, hit[c][same == 1].mean() * 100, hit[c][same == 0].mean() * 100))
