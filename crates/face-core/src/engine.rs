use std::{fs, path::Path};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{embedding::Recognizer, picture, yunet::Detector};

pub struct EngineError {
    pub code: &'static str,
    pub message: String,
}

impl EngineError {
    pub(crate) fn invalid_image(error: impl ToString) -> Self {
        Self {
            code: "INVALID_IMAGE",
            message: error.to_string(),
        }
    }

    pub(crate) fn limit(message: &str) -> Self {
        Self {
            code: "LIMIT_EXCEEDED",
            message: message.to_owned(),
        }
    }

    pub(crate) fn incompatible(error: impl ToString) -> Self {
        Self {
            code: "MODEL_INCOMPATIBLE",
            message: error.to_string(),
        }
    }

    pub(crate) fn incompatible_embedding(message: &str) -> Self {
        Self {
            code: "INCOMPATIBLE_EMBEDDING",
            message: message.to_owned(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

pub struct Detection {
    pub bounds: Rect,
    pub score: f32,
    pub landmarks: [Point; 5],
}

pub struct Embedding {
    pub space_id: String,
    pub values: Vec<f32>,
}

pub struct Face {
    pub detection: Detection,
    pub embedding: Embedding,
}

pub enum ImageInput<'a> {
    Path(&'a Path),
    Encoded(&'a [u8]),
    Rgb {
        data: &'a [u8],
        width: u32,
        height: u32,
    },
}

#[derive(Deserialize)]
struct Manifest {
    version: u32,
    detector: Model,
    recognizer: Option<Model>,
}

#[derive(Deserialize)]
struct Model {
    adapter: String,
    path: String,
    sha256: String,
}

impl Model {
    fn read(&self, directory: &Path) -> Result<Vec<u8>, EngineError> {
        let path = directory.join(&self.path);
        let data = fs::read(&path).map_err(EngineError::incompatible)?;
        let digest = format!("{:x}", Sha256::digest(&data));
        if digest != self.sha256 {
            return Err(EngineError::incompatible(format!(
                "Model SHA-256 does not match the manifest: {}",
                path.display()
            )));
        }
        Ok(data)
    }
}

pub struct Engine {
    detector: Detector,
    recognizer: Option<Recognizer>,
}

impl Engine {
    pub fn open(manifest_path: &Path, threads: usize) -> Result<Self, EngineError> {
        let json = fs::read(manifest_path).map_err(EngineError::incompatible)?;
        let manifest: Manifest =
            serde_json::from_slice(&json).map_err(EngineError::incompatible)?;
        if manifest.version != 1 {
            return Err(EngineError::incompatible("Unsupported manifest version"));
        }

        let directory = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let detector = Detector::open(
            &manifest.detector.adapter,
            &manifest.detector.read(directory)?,
            threads,
        )?;
        let recognizer = manifest
            .recognizer
            .map(|model| {
                Recognizer::open(
                    &model.adapter,
                    &model.read(directory)?,
                    &model.sha256,
                    threads,
                )
            })
            .transpose()?;
        Ok(Self {
            detector,
            recognizer,
        })
    }

    pub fn detect(&mut self, input: ImageInput<'_>) -> Result<Vec<Detection>, EngineError> {
        self.detector.detect(&picture::load(input)?)
    }

    pub fn analyze(&mut self, input: ImageInput<'_>) -> Result<Vec<Face>, EngineError> {
        let recognizer = self
            .recognizer
            .as_mut()
            .ok_or_else(|| EngineError::incompatible("The manifest has no recognizer"))?;
        let image = picture::load(input)?;
        let detections = self.detector.detect(&image)?;
        let mut faces = Vec::with_capacity(detections.len());
        for detection in detections {
            if let Some(embedding) = recognizer.embed(&image, detection.landmarks)? {
                faces.push(Face {
                    detection,
                    embedding,
                });
            }
        }
        Ok(faces)
    }
}
