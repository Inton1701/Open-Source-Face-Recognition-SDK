# One-time download of the models used by face_dlib/dlib_face.py (GitHub and Google Storage only).
# Run on a machine with internet, then copy face_dlib/models/ to offline machines.
# Licences: dlib ResNet recogniser and 5-point predictor are public domain / CC0 1.0 (github.com/davisking/dlib-models);
# MediaPipe BlazeFace full-range is Apache-2.0 (its model card on storage.googleapis.com/mediapipe-assets).
# Usage: python face_dlib/download_models.py
import bz2, hashlib, os, urllib.request
from dlib_face import MODEL_DIR

DLIB = 'https://raw.githubusercontent.com/davisking/dlib-models/master/'
FILES = {  # file: (url, sha256 of the downloaded file)
    'dlib_face_recognition_resnet_model_v1.dat': (DLIB + 'dlib_face_recognition_resnet_model_v1.dat.bz2',
                                                  'abb1f61041e434465855ce81c2bd546e830d28bcbed8d27ffbe5bb408b11553a'),
    'shape_predictor_5_face_landmarks.dat': (DLIB + 'shape_predictor_5_face_landmarks.dat.bz2',
                                             '6e787bbebf5c9efdb793f6cd1f023230c4413306605f24f299f12869f95aa472'),
    'LICENSE-dlib-models.txt': (DLIB + 'LICENSE', '6a1ee543e5282cd9061881edf462e6fdab181f328da71fc2c9a6950a80e94d01'),
    'face_detection_full_range.tflite': ('https://storage.googleapis.com/mediapipe-assets/face_detection_full_range.tflite',
                                         '3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b'),
}

os.makedirs(MODEL_DIR, exist_ok=True)
for name, (url, sha) in FILES.items():
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        print(name, 'already present'); continue
    print('downloading', name)
    data = urllib.request.urlopen(url).read()
    if hashlib.sha256(data).hexdigest() != sha:
        raise SystemExit('checksum mismatch for %s; not saved' % url)
    with open(path, 'wb') as f:
        f.write(bz2.decompress(data) if url.endswith('.bz2') else data)
    print(name, 'ready')
