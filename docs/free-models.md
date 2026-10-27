# Face models that are free for commercial use

Desk research from 2026-10-27, done in case this project ever becomes a product. As a non-commercial project it can keep using InsightFace. Every licence below was read at a primary source (LICENSE file, model card, official README). Each candidate that claimed commercial use was then checked by two independent skeptics: one read the licence files, the other looked at training data and author restrictions. A licence that is not stated explicitly counts as "not allowed". This is not legal advice, and the Philippine privacy part could not be checked at primary sources (see the end).

Scope: 182 candidates found (178 after merging duplicates). 48 claimed commercial use, and 30 of those survived both skeptics. Hugging Face, figshare, dl.fbaipublicfiles.com, dlib.net, storage.openvinotoolkit.org and privacy.gov.ph were unreachable from the research machine. Anything known only through those hosts is marked as such.

## Bottom line

- There is **no fully clean, ready-made recogniser**. Every survivor has a permissive weights licence, but its training data is research-only, scraped or undisclosed. No term was found that makes the data's restriction bind the model, so this is a risk for a lawyer to weigh, not a green light ("tier B" below).
- **InsightFace is the line not to cross.** Its README says "The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only" ([source](https://raw.githubusercontent.com/deepinsight/insightface/master/README.md)). That clause also rules out OpenCV YuNet, libfacedetection, OpenCV SFace and most RetinaFace ports, because they were trained on InsightFace's RetinaFace/SCRFD landmark labels.
- **The Faceplugin code and weights in this fork have no licence at all** (upstream has no LICENSE file), and `tuned_head.npz` inherits that. They can't go into a product without written permission from Faceplugin.
- Fine-tuning, distilling, quantising or converting a model does not lift a non-commercial licence. Only training from scratch on data you own resets provenance.

## Recognisers that survived

Tier A = weights licence permissive and training data clean. Tier B = weights licence permissive, training data research-only or unknown.

| Rank | Model | Weights licence (code licence) | Tier | Published quality | Size | Evidence |
|---|---|---|---|---|---|---|
| 1 | Intel OpenVINO face-reidentification-retail-0095 (MobileFaceNet-SE, 256-d) | Apache-2.0 via `model.yml` `license:` field (code Apache-2.0); OMZ README adds a human-rights use condition | B: training data not named; Intel's training config says VGGFace2 | LFW 99.47% | 4.4 MB FP32 | [model.yml](https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/models/intel/face-reidentification-retail-0095/model.yml) |
| 2 | dlib ResNet v1 (128-d) | public domain / CC0 1.0 (dlib library: Boost-1.0) | B: about half of the 3M images are from VGG Face and FaceScrub | LFW 99.13% single pass, 99.38% with 100-crop jitter | 21 MB | [dlib-models README](https://raw.githubusercontent.com/davisking/dlib-models/master/README.md) |
| 3 | fal AuraFace-v1 glintr100 (ResNet-100, 512-d) | Apache-2.0 per model card, read via GitHub mirrors only | B*: training data not named; the same repo ships InsightFace's SCRFD under the Apache label, so the label is unreliable | LFW 99.65% | 261 MB, Hugging Face only | [card mirror](https://raw.githubusercontent.com/maniegokryz-prog/EndDev/HEAD/faceid/models/auraface/README.md) |
| 4 | SeetaFace6 recognisers (light / ResNet-50 / mask) | no weights licence file; README: "free for commercial and personal use" (code BSD-2) | B: data undisclosed; encrypted `.csta` format, can't be fine-tuned | none published | n/a, Dropbox/Baidu only | [README](https://github.com/SeetaFace6Open/index/blob/master/README.md) |
| 5 | Meta DINOv2 ViT-S/14 (a generic backbone to fine-tune, not a face model) | Apache-2.0 for code and weights since 2023-08-31 | B: LVD-142M composition undisclosed | no face numbers | ~85 MB | [README](https://raw.githubusercontent.com/facebookresearch/dinov2/main/README.md) |
| 6 | dlib DenseNet v1 (Cydral BAREL) | MIT ("trained models are licensed under the MIT License") | B | LFW 96.1% (10-crop) | 2.5 MB | [BAREL README](https://github.com/Cydral/BAREL/blob/main/README.md) |
| 7 | CMU OpenFace nn4.small2.v1 | Apache-2.0, explicitly covering the trained models | B: FaceScrub + CASIA-WebFace | LFW 92.9% | 15 MB | [README](https://github.com/cmusatyalab/openface/blob/master/README.md) |

## Detectors that survived

| Model | Weights licence | Tier | Landmarks | Small faces | Evidence |
|---|---|---|---|---|---|
| MediaPipe BlazeFace full-range (dense / sparse) | Apache-2.0 (model card) | A: Google's own consented images | 6 rough points (eyes, nose tip, mouth centre, ears), no mouth corners | weak: faces must be at least 5% of the frame side; card says surveillance is out of scope | [model card PDF](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Full%20Range).pdf) |
| MediaPipe BlazeFace short-range | Apache-2.0 | A | same | very poor for CCTV (selfie range) | [model card PDF](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf) |
| Intel face-detection-0205 (+ landmarks-regression-retail-0009 for 5 points) | Apache-2.0 (`model.yml`) | B: WIDER FACE (landmarks model: A-, Intel internal data) | none; the separate 0009 model gives 5 points | WIDER hard 54-56 | [model.yml](https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/models/intel/face-detection-0205/model.yml) |
| Intel face-detection-0206 (ResNet-152, heavy) | Apache-2.0 | B | none | WIDER hard 84.4 | [model.yml](https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/models/intel/face-detection-0206/model.yml) |
| Intel face-detection-0200/0202/0204, retail-0004/0005, adas-0001 | Apache-2.0 | B | none | poor (hard 37-52) | [model.yml](https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/models/intel/face-detection-retail-0004/model.yml) |
| CenterFace | MIT ("Code and models are all MIT", maintainer in issue #39) | B, high risk: maintainer says the landmark labels may be InsightFace's (issue #52) | 5 points | best here: WIDER hard 78.2 | [issue #39](https://github.com/Star-Clouds/CenterFace/issues/39) |
| SeetaFace6 detector + 5-point landmarker | README commercial statement only | B | 5 points | not published | [README](https://github.com/SeetaFace6Open/index/blob/master/README.md) |

## Dropped, and why

- **InsightFace weights** (buffalo_s/l, SCRFD, ArcFace zoo, antelopev2, UniFace/deepface/Hailo re-hosts): non-commercial.
- **OpenCV Zoo SFace**: refuted. The Apache label names no copyright holder, upstream zhongyy/SFace has no licence, and its documented training data comes from the InsightFace data zoo.
- **OpenCV Zoo YuNet, libfacedetection weights, Ultra-Light-Fast-Generic-Face-Detector, biubug6 RetinaFace, Rockchip RetinaFace**: trained on WIDER FACE with InsightFace RetinaFace/SCRFD labels, which are non-commercial "and the models trained with these data". The Ultra-Light weights are the ones this fork's Faceplugin detector ships.
- **AdaFace, CVLface, EdgeFace, GhostFaceNets, TransFace, LVFace, FaceX-Zoo, TFace, cavaface, PaddlePaddle, community MobileFaceNets, facenet-pytorch, deepface conversions**: no weights licence stated, or explicitly non-commercial (EdgeFace's Hugging Face cards say CC BY-NC-SA 4.0; LVFace says research only), and mostly trained on MS1M/WebFace/VGGFace2.
- **YOLO-face family**: no weights licence, and GPL-3.0/AGPL-3.0 (Ultralytics) lineage.
- **Tencent DSFD**: academic research only. **NVIDIA TAO FaceDetect**: proprietary EULA. **Qualcomm AI Hub face models**: separate Qualcomm licence and research-only training data.
- **dlib 68-point predictor and mmod detector**: the dlib-models README says the 68-point model "can't be used in a commercial product" (iBUG 300-W); the mmod detector's training data includes non-commercial sets. Only the 5-point predictor is clean.
- **Synthetic-face datasets and generators** (DigiFace-1M, Microsoft Face Synthetics, SynFace, DCFace, IDiff-Face, Vec2Face, Arc2Face, HyperFace, GANDiffFace, SFace/SFace2, SFHQ, StyleGAN FFHQ models): non-commercial (DigiFace-1M's R-UDA counts trained models as "Results"), unlicensed, or built on FFHQ/WebFace/InsightFace.
- **Real-photo identity datasets** (MS1M, Glint360K, WebFace260M family, CASIA-WebFace, VGGFace2, CelebA, FFHQ, UMDFaces, WIDER FACE): non-commercial, withdrawn or unverifiable.

## Training data and code you could use

- **Code** (permissive, train from random init only): CVLface (MIT), InsightFace arcface_torch/partial_fc (MIT, README grant only), SCRFD training code (Apache-2.0), FaceX-Zoo and TFace code (Apache-2.0, excluding TFace `generation/uiface`), libfacedetection.train (BSD-3), Intel OpenVINO Training Extensions legacy recipes (Apache-2.0), Cydral BAREL (MIT).
- **Detector data**: Open Images V7 "Human face" boxes (annotations CC BY 4.0, images CC BY 2.0, with attribution per image; use the 95,335 exhaustively labelled images). No open landmark labels are clean, so new 5-point labels have to be made: hand-label a seed set, then pseudo-label with a permissive model such as Intel landmarks-regression-retail-0009.
- **Identity data**: no open identity-labelled face dataset is commercially clean. The realistic route is your own consented set: staff and paid volunteers, with a written release and consent specific to model training.

## How to make a model truly "ours"

1. Start from a base whose weights licence allows modification and rebranding: Intel face-reidentification-retail-0095 (Apache-2.0, cheapest on CPU, Intel's training code is Apache-2.0) or dlib ResNet (CC0).
2. Fine-tune on consented photos captured on your own cameras (day, night/IR, masks, distance), aiming for 1,000+ people x 20-50 crops each, with a training-specific consent separate from attendance enrolment. Enrolment photos from customers are not training data unless each person separately opts in.
3. Never use a non-commercial model anywhere in the pipeline: not as teacher, labeller, face cropper or quality filter.
4. Fine-tuning keeps the base model's provenance (VGGFace2 for Intel, VGG Face/FaceScrub for dlib). Record it in a provenance sheet: file, SHA-256, base model, licence URL, training data, who trained it. Have counsel review it.
5. Rebranding the product is fine. Keep the upstream notices in a THIRD-PARTY-NOTICES file and mark fine-tuned weights as modified (Apache-2.0 section 4b). Don't use Intel, OpenVINO, MediaPipe, Google, Meta, dlib or fal in the product name.

## Notice duties when rebranding (closed-source product)

| Licence | What to ship |
|---|---|
| Apache-2.0 (Intel OMZ, MediaPipe, DINOv2, OpenFace, OpenVINO, OpenCV) | Licence text and copyright headers, plus a prominent "modified" note on fine-tuned or converted weights. No trademark rights (section 6). Patent grant included. |
| MIT (BAREL, CenterFace, onnxruntime, InsightFace code) | Copyright line and permission text |
| BSD-2/3 (SeetaFace, libfacedetection.train, PyTorch) | Notice and disclaimer in the docs; BSD-3 bars endorsement using contributors' names |
| Boost-1.0 (dlib library) | Notice in source copies (not needed for object code only) |
| CC0 (dlib models) | Nothing required; keep a provenance note. No trademark or patent waiver. |
| LGPL (FFmpeg/Qt inside opencv-python) | Use `opencv-python-headless`, or meet the relinking duties |

Use conditions that aren't licence terms but still matter: the Intel OMZ README condition (no products that contribute to human-rights violations), and the MediaPipe model card ("Any form of surveillance or identity recognition is explicitly out of scope").

## Philippine Data Privacy Act (RA 10173): NOT verified

privacy.gov.ph was unreachable and the search budget ran out, so treat these as conservative design assumptions to check with counsel and the NPC's own circulars, not findings:

- Treat face templates and enrolment photos as sensitive personal information, and rely on consent. Offer a non-biometric fallback (RFID or PIN) so consent can really be refused. Minors need a parent or guardian flow.
- Keep attendance recognition separate from security CCTV, and discard templates of people who aren't enrolled. Put up signage and a privacy notice.
- Do a privacy impact assessment and set separate retention periods for frames, templates and attendance logs. Have a breach process (NPC Circular 16-03, 72 hours) and check whether NPC registration applies (Circular 2022-04).

## Benchmark against InsightFace

`face_dlib/` is the second backend: dlib ResNet v1 recogniser (CC0) plus MediaPipe BlazeFace full-range (Apache-2.0, run with LiteRT) and dlib's CC0 5-point predictor. It has the same `get(img) -> count, boxes, scores, aligned, embeddings` interface as `face_insight/`. It is the top pair whose weights are on GitHub/Google Storage. The Intel trio needs storage.openvinotoolkit.org, and AuraFace needs Hugging Face. LFW results are still pending (figshare blocked on the research machine); the commands are in the README.
