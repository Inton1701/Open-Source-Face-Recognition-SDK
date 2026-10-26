# One-time download of the InsightFace model packs used by face_insight/insight.py.
# Run on a machine with internet, then copy face_insight/models/ to offline machines.
# The models are for non-commercial research use only (InsightFace licence).
# Usage: python face_insight/download_models.py [s] [l]   (default: both)
import os, sys, urllib.request, zipfile
from insight import MODEL_DIR, PACKS

URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7/%s.zip'

for size in sys.argv[1:] or ['s', 'l']:
    pack, det, rec = PACKS[size]
    if all(os.path.exists(os.path.join(MODEL_DIR, pack, f)) for f in (det, rec)):
        print(pack, 'already present'); continue
    os.makedirs(os.path.join(MODEL_DIR, pack), exist_ok=True)
    zpath = os.path.join(MODEL_DIR, pack + '.zip')
    print('downloading', pack)
    urllib.request.urlretrieve(URL % pack, zpath)
    with zipfile.ZipFile(zpath) as z:
        for f in (det, rec):  # only the detector and recogniser are used
            member = next(n for n in z.namelist() if n.endswith(f))
            with z.open(member) as src, open(os.path.join(MODEL_DIR, pack, f), 'wb') as dst:
                dst.write(src.read())
    os.remove(zpath)
    print(pack, 'ready')
