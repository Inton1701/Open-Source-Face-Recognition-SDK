# Train the optional linear head on LFW embeddings and save it to face_feature/tuned_head.npz.
# Usage: python eval/finetune_head.py <lfw_dir>   (run eval/embed_lfw.py first)
import os, sys
import numpy as np
import torch

def train_head(F, a, b, y, steps=300):
    """Learn W (starting at identity) so cosine(F@W) separates same/different pairs."""
    torch.manual_seed(0)
    Ft = torch.tensor(F)
    eye = torch.eye(F.shape[1])
    W = torch.nn.Parameter(eye.clone())
    scale = torch.nn.Parameter(torch.tensor(10.0))
    bias = torch.nn.Parameter(torch.tensor(-5.0))
    opt = torch.optim.Adam([W, scale, bias], lr=1e-3)
    ia, ib, yt = torch.tensor(a), torch.tensor(b), torch.tensor(y, dtype=torch.float32)
    for _ in range(steps):
        Z = torch.nn.functional.normalize(Ft @ W, dim=1)
        cos = (Z[ia] * Z[ib]).sum(1)
        # pull W toward identity so a small pair set cannot overfit it
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scale * cos + bias, yt) + 1e-2 * ((W - eye) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach().numpy()

def project(F, W):
    Z = F @ W
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)

def best_threshold(scores, y):
    ts = np.unique(scores)
    return float(ts[np.argmax([((scores > t) == y).mean() for t in ts])])

if __name__ == '__main__':
    d = np.load(os.path.join(sys.argv[1], 'embeddings.npz'))
    F, found, P = d['F'], d['found'], d['P']
    fold, same, a, b = P.T
    ok = found[a] & found[b]
    W = train_head(F, a[ok], b[ok], same[ok])
    Z = project(F, W)
    scores = (np.sum(Z[a[ok]] * Z[b[ok]], 1) + 1) * 50
    t = best_threshold(scores, same[ok])
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'face_feature', 'tuned_head.npz')
    np.savez(out, W=W.astype(np.float32), threshold=np.float32(t))
    print('saved %s (threshold %.2f, trained on %d pairs)' % (out, t, ok.sum()))
