# Synthetic ONNX fixtures

These two graphs were generated for this project with ONNX 1.23.2, IR version 8, and opset 17. They contain only arithmetic constants, with no training data or pretrained model weights.

- `detector.onnx` has the input and output shapes expected by the SCRFD adapter. It emits one fixed bounding box and five landmarks when the mean of the input tensor is positive. Otherwise it emits no detections.
- `recognizer.onnx` has the input and output shapes expected by the ArcFace adapter. It emits a vector of 512 ones so the installed-package test can check normalization and cosine comparison.

`manifest.json` pins both files by SHA-256. The fixtures exercise the engine and packaging in CI. They are intended for integration testing.
