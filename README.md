# @jrmc/face-engine

Local face detection and embedding generation for Node.js. `FaceEngine` accepts a JPEG or PNG path, encoded image bytes, or an RGB frame. `detect()` returns a bounding box, a detection score, and five landmarks per face in the orientation-corrected image. `analyze()` also returns a normalized embedding when a recognition model is configured.

## Project layout

| Path | Purpose |
| --- | --- |
| `crates/face-core/` | Rust engine. Decodes images, detects faces with SCRFD, aligns them, and generates embeddings with ONNX Runtime. |
| `crates/face-napi/` | NAPI-RS binding. Exposes the Rust engine and converts its results into JavaScript objects. |
| `src/index.ts` | Public TypeScript API. Starts the engine, manages the bounded request queue, and handles shutdown. |
| `src/detection-worker.ts` | Node.js worker that owns the Rust engine and processes requests off the main thread. |
| `scripts/` | Development tools for checking ONNX graphs. |
| `tests/` | Node.js integration tests for the engine API. |
| `profiles/` | Versioned manifests that specify the ONNX files and their SHA-256 hashes. |
| `models/`, `photos/` | Local ONNX weights and test photos. Git ignores both directories. |

The Cargo and npm manifests and lockfiles are at the repository root. Cargo builds into `target/`. NAPI-RS writes the `.node` addon, its ESM loader, and its declarations to `build/`. TypeScript compiles the sources, scripts, and tests into `dist/`. The npm entry point is `dist/src/index.js`, with declarations at `dist/src/index.d.ts`. Git ignores these generated directories.

## Development

Building from source requires Node.js 24, Rust 1.92 or later, a C/C++ toolchain, and OpenSSL development headers.

```sh
npm ci
npm run build
npm run typecheck
```

The integration tests and `npm run check:models` use `det_500m.onnx` and `w600k_mbf.onnx` in `models/buffalo_sc/`. The sample profile verifies the SHA-256 hash of each model. InsightFace provides these weights for non-commercial research. [Model release](https://github.com/deepinsight/insightface/releases/tag/model-zoo) · [Model license](https://github.com/deepinsight/insightface/blob/master/server/backend/insightface_server/licensing/defaults/buffalo_sc/MODEL.LICENSE)

```sh
mkdir -p models/buffalo_sc
curl -fL https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_sc.zip \
  -o models/buffalo_sc.zip
printf '%s\n' '57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72  models/buffalo_sc.zip' \
  | sha256sum --check && unzip -n models/buffalo_sc.zip -d models/buffalo_sc
npm test
npm run check:models
```

## Analyze an image

With the [`buffalo-sc` profile](profiles/buffalo-sc.json), each embedding is a 512-value `Float32Array` with a `spaceId` that identifies the model weights and alignment recipe. A manifest can omit `recognizer` for detection-only use. Calling `analyze()` with that manifest rejects with `MODEL_INCOMPATIBLE`.

```js
import { FaceEngine } from './dist/src/index.js'

const engine = await FaceEngine.create({ manifest: './profiles/buffalo-sc.json' })
try {
  const faces = await engine.analyze('./photos/portrait.jpg')
  console.log(faces.length)
} finally {
  await engine.close()
}
```
