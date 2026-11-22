# @jrmc/face-engine

Local face detection and embedding generation for Node.js. `FaceEngine` accepts a JPEG or PNG path, encoded image bytes, or an RGB frame. `detect()` returns a bounding box, a detection score, and five landmarks per face in the orientation-corrected image. `analyze()` also returns a normalized embedding when a recognition model is configured.

Face Engine's code is licensed under the [MIT License](LICENSE). The separately downloaded [model weights](licenses/models/README.md) retain their upstream licenses.

## Install

```sh
npm install @jrmc/face-engine
```

The npm package selects its native addon through optional dependencies. See the [usage guide](docs/usage.md) for setting up the models, processing photos, and storing embeddings in your application.

## Project layout

| Path | Purpose |
| --- | --- |
| `crates/face-core/` | Rust engine. Decodes images, detects faces with YuNet, aligns them, and generates embeddings with ONNX Runtime. |
| `crates/face-napi/` | NAPI-RS binding. Exposes the Rust engine and converts its results into JavaScript objects. |
| `src/index.ts` | Public TypeScript API. Starts the engine, manages the bounded request queue, and handles shutdown. |
| `src/detection-worker.ts` | Node.js worker that owns the Rust engine and processes requests off the main thread. |
| `scripts/` | Development tools for checking ONNX graphs, comparing photos, benchmarking, and packaging. |
| `tests/` | Node.js integration tests for the engine API. |
| `profiles/` | Versioned manifests that specify the ONNX files and their SHA-256 hashes. |
| `models/`, `photos/` | Local ONNX weights and test photos. Git ignores both directories. |

The Cargo and npm manifests and lockfiles are at the repository root. Cargo builds into `target/`. NAPI-RS writes the `.node` addon, its ESM loader, and its declarations to `build/`. TypeScript compiles the sources, scripts, and tests into `dist/`. The npm entry point is `dist/src/index.js`, with declarations at `dist/src/index.d.ts`. Git ignores these generated directories and the local archives in `artifacts/`.

## Development

Building from source requires Node.js 24, Rust 1.92 or later, a C/C++ toolchain, and OpenSSL development headers.

```sh
npm ci
npm run build
npm run typecheck
```

## Model setup

Download the original YuNet and SFace ONNX weights from OpenCV Zoo. The [profile](profiles/yunet-sface.json) pins both SHA-256 hashes.

```sh
mkdir -p models/yunet-sface
curl -fL https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx \
  -o models/yunet-sface/face_detection_yunet_2023mar.onnx
curl -fL https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx \
  -o models/yunet-sface/face_recognition_sface_2021dec.onnx
npm run check:models
npm test
```

The weights are supplied separately from the npm packages. See the [model licenses and attribution](licenses/models/README.md) for redistribution conditions.

Run the integration tests without downloading the weights using the synthetic graphs bundled with the source tree.

```sh
FACE_ENGINE_TEST_MANIFEST=tests/fixtures/manifest.json npm test
```

## Analyze an image

The [`yunet-sface` profile](profiles/yunet-sface.json) returns a 128-value `Float32Array` and a `spaceId` that identifies the model weights and alignment recipe. A manifest can omit `recognizer` for detection-only use. Calling `analyze()` with that manifest rejects with `MODEL_INCOMPATIBLE`.

```js
import { FaceEngine } from '@jrmc/face-engine'

const engine = await FaceEngine.create({ manifest: './profiles/yunet-sface.json' })
try {
  const faces = await engine.analyze('./photos/portrait.jpg')
  console.log(faces.length)
} finally {
  await engine.close()
}
```

## Compare two portraits

```sh
npm run compare -- photos/portrait-a.jpg photos/portrait-b.jpg
```

Each photo must contain exactly one analyzable face. The script prints a cosine similarity score between −1 and 1. Pass a third argument to select another manifest. `engine.compare(a, b)` is synchronous and throws `INCOMPATIBLE_EMBEDDING` for different embedding spaces, invalid vectors, or a zero norm. The calling application sets its own identity threshold.

## Benchmarks and local installation

`npm run --silent bench -- photos 400` runs 400 analyses at each of 1, 2, and 4 threads, using a separate process for each setting. It reports latency, throughput, and process memory usage.

Local archives package the ESM API and its declarations separately from the Linux x64 addon. The application supplies the model files. The same x64 archives passed offline installation and runtime tests on Ubuntu 24.04 and Debian 13.7 with Node.js 24.22.1. The current addon requires `GLIBC_2.38` and `GLIBCXX_3.4.31`. Debian 12 ships older libraries and is incompatible with this build.

Install both archives from a consuming project.

```sh
npm install --offline --ignore-scripts \
  /path/to/artifacts/jrmc-face-engine-0.1.0.tgz \
  /path/to/artifacts/jrmc-face-engine-linux-x64-gnu-0.1.0.tgz
```

The [Native packages workflow](.github/workflows/native-packages.yml) builds Linux x64 and ARM64 archives. Each job installs the archives offline in fresh Ubuntu 24.04 and Debian 13 containers, then tests detection, embedding generation, and comparison with synthetic graphs. The `.tgz` files are available as workflow artifacts after the checks pass.

## Photo library example

The example uses these archives in a small Node.js and SQLite application for importing photos and sorting faces by person. Follow the [model setup](#model-setup) and the [example setup instructions](example/README.md) to prepare the app. Then start it from the repository root.

```sh
npm --prefix example start
```

Open [http://127.0.0.1:4173](http://127.0.0.1:4173) to use the photo library.

To override the port or data directory, set the environment variables when starting the example.

```sh
FACE_ENGINE_MANIFEST="$PWD/profiles/yunet-sface.json" \
FACE_ENGINE_DATA_DIR="$PWD/example/data" \
PORT=4175 npm --prefix example start
```
