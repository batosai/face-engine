use std::{fs, path::Path};

use image::ImageBuffer;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{picture, scrfd::Detector};

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

#[derive(Deserialize)]
struct Manifest {
    version: u32,
    detector: Model,
}

#[derive(Deserialize)]
struct Model {
    adapter: String,
    path: String,
    sha256: String,
}

pub struct Engine {
    detector: Detector,
}

impl Engine {
    pub fn open(manifest_path: &Path, threads: usize) -> Result<Self, EngineError> {
        let json = fs::read(manifest_path).map_err(EngineError::incompatible)?;
        let manifest: Manifest =
            serde_json::from_slice(&json).map_err(EngineError::incompatible)?;
        if manifest.version != 1 || manifest.detector.adapter != "scrfd-640-v1" {
            return Err(EngineError::incompatible(
                "Unsupported manifest version or detector adapter",
            ));
        }

        let directory = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let model =
            fs::read(directory.join(&manifest.detector.path)).map_err(EngineError::incompatible)?;
        let digest = format!("{:x}", Sha256::digest(&model));
        if digest != manifest.detector.sha256 {
            return Err(EngineError::incompatible(
                "Detector SHA-256 does not match the manifest",
            ));
        }

        Ok(Self {
            detector: Detector::open(&model, threads)?,
        })
    }

    pub fn detect_path(&mut self, path: &Path) -> Result<Vec<Detection>, EngineError> {
        let encoded = picture::read_file(path)?;
        self.detect_encoded(&encoded)
    }

    pub fn detect_encoded(&mut self, encoded: &[u8]) -> Result<Vec<Detection>, EngineError> {
        self.detector.detect(&picture::decode(encoded)?)
    }

    pub fn detect_rgb(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<Detection>, EngineError> {
        picture::validate_dimensions(width, height)?;
        if data.len() != width as usize * height as usize * 3 {
            return Err(EngineError::invalid_image(
                "rgb24 length must equal width × height × 3",
            ));
        }
        let image = ImageBuffer::from_raw(width, height, data)
            .ok_or_else(|| EngineError::invalid_image("Invalid rgb24 image"))?;
        self.detector.detect(&image)
    }
}
