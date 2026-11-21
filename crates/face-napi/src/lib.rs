use std::path::Path;

use face_core::ImageInput;
use napi::bindgen_prelude::{Float32Array, Uint8Array};
use napi_derive::napi;

#[napi(object)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[napi(object)]
pub struct Rectangle {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[napi(object)]
pub struct Detection {
    pub r#box: Rectangle,
    pub detection_score: f64,
    #[napi(ts_type = "[Point, Point, Point, Point, Point]")]
    pub landmarks: Vec<Point>,
}

#[napi(object)]
pub struct Embedding {
    pub space_id: String,
    pub values: Float32Array,
}

#[napi(object)]
pub struct Face {
    pub r#box: Rectangle,
    pub detection_score: f64,
    #[napi(ts_type = "[Point, Point, Point, Point, Point]")]
    pub landmarks: Vec<Point>,
    pub embedding: Embedding,
}

#[napi]
pub struct NativeDetector(Option<face_core::Engine>);

#[napi]
impl NativeDetector {
    #[napi(factory)]
    pub fn open(manifest: String, threads: u32) -> napi::Result<Self> {
        face_core::Engine::open(Path::new(&manifest), threads as usize)
            .map(|engine| Self(Some(engine)))
            .map_err(js_error)
    }

    #[napi]
    pub fn detect_path(&mut self, path: String) -> napi::Result<Vec<Detection>> {
        self.detect(ImageInput::Path(Path::new(&path)))
    }

    #[napi]
    pub fn detect_encoded(&mut self, data: Uint8Array) -> napi::Result<Vec<Detection>> {
        self.detect(ImageInput::Encoded(&data))
    }

    #[napi]
    pub fn detect_rgb(
        &mut self,
        data: Uint8Array,
        width: u32,
        height: u32,
    ) -> napi::Result<Vec<Detection>> {
        self.detect(ImageInput::Rgb {
            data: &data,
            width,
            height,
        })
    }

    #[napi]
    pub fn analyze_path(&mut self, path: String) -> napi::Result<Vec<Face>> {
        self.analyze(ImageInput::Path(Path::new(&path)))
    }

    #[napi]
    pub fn analyze_encoded(&mut self, data: Uint8Array) -> napi::Result<Vec<Face>> {
        self.analyze(ImageInput::Encoded(&data))
    }

    #[napi]
    pub fn analyze_rgb(
        &mut self,
        data: Uint8Array,
        width: u32,
        height: u32,
    ) -> napi::Result<Vec<Face>> {
        self.analyze(ImageInput::Rgb {
            data: &data,
            width,
            height,
        })
    }

    #[napi]
    pub fn close(&mut self) {
        self.0 = None;
    }
}

impl NativeDetector {
    fn engine(&mut self) -> napi::Result<&mut face_core::Engine> {
        self.0
            .as_mut()
            .ok_or_else(|| napi::Error::from_reason("ENGINE_CLOSED: The engine is closed"))
    }

    fn detect(&mut self, input: ImageInput<'_>) -> napi::Result<Vec<Detection>> {
        Ok(self
            .engine()?
            .detect(input)
            .map_err(js_error)?
            .into_iter()
            .map(Detection::from)
            .collect())
    }

    fn analyze(&mut self, input: ImageInput<'_>) -> napi::Result<Vec<Face>> {
        Ok(self
            .engine()?
            .analyze(input)
            .map_err(js_error)?
            .into_iter()
            .map(Face::from)
            .collect())
    }
}

fn js_error(error: face_core::EngineError) -> napi::Error {
    napi::Error::from_reason(format!("{}: {}", error.code, error.message))
}

impl From<face_core::Detection> for Detection {
    fn from(face: face_core::Detection) -> Self {
        Self {
            r#box: Rectangle {
                x: f64::from(face.bounds.x),
                y: f64::from(face.bounds.y),
                width: f64::from(face.bounds.width),
                height: f64::from(face.bounds.height),
            },
            detection_score: f64::from(face.score),
            landmarks: face
                .landmarks
                .into_iter()
                .map(|point| Point {
                    x: f64::from(point.x),
                    y: f64::from(point.y),
                })
                .collect(),
        }
    }
}

impl From<face_core::Face> for Face {
    fn from(face: face_core::Face) -> Self {
        let detection = Detection::from(face.detection);
        Self {
            r#box: detection.r#box,
            detection_score: detection.detection_score,
            landmarks: detection.landmarks,
            embedding: Embedding {
                space_id: face.embedding.space_id,
                values: Float32Array::new(face.embedding.values),
            },
        }
    }
}

#[napi(object)]
pub struct GraphInfo {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl From<face_core::GraphInfo> for GraphInfo {
    fn from(graph: face_core::GraphInfo) -> Self {
        Self {
            inputs: graph.inputs,
            outputs: graph.outputs,
        }
    }
}

#[napi(object)]
pub struct ModelProbe {
    pub runtime: String,
    pub detector: GraphInfo,
    pub recognizer: GraphInfo,
}

#[napi]
pub fn probe_models(detector: String, recognizer: String) -> napi::Result<ModelProbe> {
    let models = face_core::probe_models(Path::new(&detector), Path::new(&recognizer))
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;

    Ok(ModelProbe {
        runtime: models.runtime.to_owned(),
        detector: models.detector.into(),
        recognizer: models.recognizer.into(),
    })
}
