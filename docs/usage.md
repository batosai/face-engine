# Use Face Engine in a Node.js application

## Install

Use Node.js 24 on Linux x64 or ARM64. The release workflow tests offline installation on Ubuntu 24.04 and Debian 13 for both architectures.

```sh
npm install @jrmc/face-engine
```

Import `FaceEngine` from `@jrmc/face-engine`. Keep npm optional dependencies enabled so it can install the native addon for your server. If the app is deployed in a container, install the dependencies for the container's Linux architecture.

## Set up the models

The included manifest pins the YuNet detector and the SFace recognizer by SHA-256. Copy the manifest into your application and download the two ONNX files. The paths in the manifest are relative to the manifest file.

```sh
mkdir -p profiles models/yunet-sface
cp node_modules/@jrmc/face-engine/profiles/yunet-sface.json profiles/
curl -fL https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx \
  -o models/yunet-sface/face_detection_yunet_2023mar.onnx
curl -fL https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx \
  -o models/yunet-sface/face_recognition_sface_2021dec.onnx
```

Make both files and the manifest available to the process that handles photo analysis. The engine opens local files and verifies the hashes on startup. YuNet is MIT-licensed and SFace is Apache 2.0-licensed. [Model attribution and redistribution terms](https://github.com/batosai/face-engine/tree/main/licenses/models)

## Analyze a photo

```ts
import { resolve } from 'node:path'
import { FaceEngine } from '@jrmc/face-engine'

const engine = await FaceEngine.create({
  manifest: resolve('profiles/yunet-sface.json'),
  threads: 1,
  maxPending: 1,
})

try {
  for (const face of await engine.analyze(resolve('photos/group.jpg'))) {
    console.log(face.box, face.detectionScore, face.embedding.spaceId)
    // Store the face geometry and the embedding with the photo ID.
  }
} finally {
  await engine.close()
}
```

`analyze()` returns one result per face and an empty array when there are no results. It accepts a JPEG or PNG path, encoded image bytes such as a Node.js `Buffer`, or a packed RGB frame of the form `{ format: 'rgb24', data, width, height }`. An RGB frame uses exactly three bytes per pixel.

Use `detect()` when you only need bounding boxes, scores, and five landmarks. Coordinates are in pixels in the orientation-corrected image. The score represents confidence in the detection.

## Store and compare faces

Save each face's geometry, photo ID, `embedding.spaceId`, and 128-value `embedding.values`. Store the person assignment as a separate field. This lets you keep manual confirmations when you regenerate the embeddings for a new model.

`engine.compare(a, b)` returns cosine similarity from −1 to 1. It accepts the `embedding` objects returned by `analyze()`. Only compare embeddings with the same `spaceId`. A PostgreSQL application using pgvector can store the values in a `vector(128)` column and filter candidates by `spaceId` before ranking them by cosine distance.

Start with faces confirmed by a person as references. Use the matches to make suggestions and measure false matches on your own photos before choosing a threshold for automatic assignments.

## Run analysis in a background worker

Reuse an engine instance across photo jobs and call `close()` when the process shuts down. Each engine processes one request at a time. `threads` controls ONNX Runtime's thread count and `maxPending` controls how many additional requests the instance can hold. Both default to 1. Keep larger backlogs in the application's job queue.

| Error code | What to check |
| --- | --- |
| `MODEL_INCOMPATIBLE` | Check the manifest, model paths and hashes, and that the Linux native dependency is installed. |
| `INVALID_IMAGE` | Check that the image is a valid JPEG, PNG, or RGB frame. |
| `LIMIT_EXCEEDED` | Check the image size or the engine options. |
| `QUEUE_FULL` | Schedule the job again when the engine has capacity. |
| `INCOMPATIBLE_EMBEDDING` | Check that both vectors have 128 finite values, nonzero norms, and matching spaces. |
| `ENGINE_CLOSED` | Create a new engine instance for subsequent jobs. |
