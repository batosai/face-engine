# Photo library example

This example uses the installed `@jrmc/face-engine` package to sort a local photo library by person. Node.js serves the interface and writes the library to SQLite through `node:sqlite`. The browser displays face crops and submits assignments.

## Set up

Use Node.js 24 on a supported Linux x64 or ARM64 system. Download the `npm-package` artifact and the `linux-x64-gnu` or `linux-arm64-gnu` artifact from a successful [Native packages build](https://github.com/batosai/face-engine/actions/workflows/native-packages.yml) and extract the `.tgz` files.

From `example/`, install the development dependencies, then install both archives. The `--no-save` flags keep the temporary archive paths out of the project manifest and lockfile.

```sh
npm ci
npm install --no-save --package-lock=false --ignore-scripts \
  /path/to/jrmc-face-engine-0.0.0.tgz \
  /path/to/jrmc-face-engine-linux-x64-gnu-0.0.0.tgz
npm run build
npm start
```

Open `http://127.0.0.1:4173`. The default model is the repository’s `buffalo-sc` profile. Follow the [model setup instructions](../README.md#development) first or set `FACE_ENGINE_MANIFEST` to your own local manifest. The application is accessible on the local machine.

| Setting | Default |
| --- | --- |
| `FACE_ENGINE_MANIFEST` | `../profiles/buffalo-sc.json` |
| `FACE_ENGINE_DATA_DIR` | `example/data/` |
| `PORT` | `4173` |

## Sort the library

- **Photos** accepts JPEG and PNG files by drag and drop or through the file picker. Open a photo to review its faces, retry analysis, or delete it.
- **To review** shows each unassigned face. Choose a suggested or existing person, or create a person while assigning the face. Suggestions rank confirmed faces in the same embedding space by cosine similarity.
- **People** groups assigned faces by person. Open an album, then a photo, to correct or clear an assignment.

The SQLite database and original files persist after shutdown. A single engine processes the queue in order. Reanalysis preserves existing assignments when the faces match the previous detections. The API returns the crops’ coordinates and suggestions while keeping embeddings in SQLite.

Run `npm test` after building to exercise upload, assignment, suggestions, replay, restart, and deletion using the repository’s synthetic graphs and temporary storage.
