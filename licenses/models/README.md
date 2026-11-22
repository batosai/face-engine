# Pretrained model licenses

Face Engine loads the original ONNX files listed below. Each file's SHA-256 is pinned in the profile. The model weights are downloaded separately from the npm packages.

| File | Upstream | License |
| --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | [YuNet in OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet#license) | [MIT](YuNet-MIT.txt), copyright 2020 Shiqi Yu |
| `face_recognition_sface_2021dec.onnx` | [SFace in OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface#license) | [Apache License 2.0](SFace-Apache-2.0.txt) |

When redistributing the YuNet weights, include the MIT license and copyright notice. When redistributing the SFace weights, include the Apache License 2.0 text and preserve the applicable upstream notices. Include the upstream `NOTICE` if one accompanies the distribution, and mark any modified files. OpenCV Zoo publishes the SFace model directory under Apache 2.0. The exact training-data provenance for this ONNX release is tracked in [OpenCV Zoo issue #313](https://github.com/opencv/opencv_zoo/issues/313).
