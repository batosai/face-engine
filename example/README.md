# Photo library example

This example uses the installed `@jrmc/face-engine` package to sort a local photo library by person. Node.js serves the interface and writes the library to SQLite through `node:sqlite`. The browser displays face crops and submits assignments.

## Set up

Use Node.js 24 on a supported Linux x64 or ARM64 system, and follow the [YuNet and SFace model setup](../README.md#model-setup). Download the `npm-package` artifact and the `linux-x64-gnu` or `linux-arm64-gnu` artifact from a successful [Native packages build](https://github.com/batosai/face-engine/actions/workflows/native-packages.yml) and extract the `.tgz` files.

From `example/`, install the development dependencies, then install both archives. The `--no-save` flags keep the temporary archive paths out of the project manifest and lockfile.

```sh
npm ci
npm install --no-save --package-lock=false --ignore-scripts \
  /path/to/jrmc-face-engine-0.1.0.tgz \
  /path/to/jrmc-face-engine-linux-x64-gnu-0.1.0.tgz
npm run build
npm start
```

Open `http://127.0.0.1:4173`. The default profile uses YuNet and SFace. The application is accessible on the local machine.

| Setting | Default |
| --- | --- |
| `FACE_ENGINE_MANIFEST` | `../profiles/yunet-sface.json` |
| `FACE_ENGINE_DATA_DIR` | `example/data/` |
| `PORT` | `4173` |

## Build from this checkout

From the repository root, build Face Engine from this checkout with `npm run build` (or `toolbox run --container face-engine npm run build`). Install the checkout in the example and build the UI.

```sh
npm --prefix example install --no-save --package-lock=false --ignore-scripts "$PWD"
npm --prefix example run build
```

## Run locally

From the repository root, start the example with `npm --prefix example start`. To use a different port, set `PORT=4175`.

The example stores its SQLite database and uploaded photos in `example/data/`. To reuse a library created with an older model, back it up first. Start the example with `FACE_ENGINE_DATA_DIR` pointing to the copy, then reanalyze each photo. Suggestions compare embeddings only after they share the same `spaceId`. Reanalysis retains existing assignments when detections can be matched safely.

## Sort the library

- **Photos** accepts JPEG and PNG files by drag and drop or through the file picker. Open a photo to review its faces, retry analysis, or delete it.
- **To review** shows each unassigned face. Choose a suggested or existing person, or create a person while assigning the face. Suggestions rank confirmed faces in the same embedding space by cosine similarity.
- **People** groups assigned faces by person. Open an album, then a photo, to correct or clear an assignment.

The SQLite database and original files persist after shutdown. A single engine processes the queue in order. Reanalysis preserves existing assignments when the faces match the previous detections. The API returns the crops’ coordinates and suggestions while keeping embeddings in SQLite.

Run `npm test` after building to exercise upload, assignment, suggestions, replay, restart, and deletion using the repository’s synthetic graphs and temporary storage.
