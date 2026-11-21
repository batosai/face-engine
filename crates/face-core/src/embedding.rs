use std::ops::Deref;

use image::{ImageBuffer, Rgb, RgbImage};
use ort::{
    session::Session,
    value::{Tensor, TensorElementType, ValueType},
};

use crate::{
    Embedding, EngineError, Point,
    align::{self, SIDE},
};

pub(crate) const ADAPTER: &str = "arcface-rgb112-v1";
const DIMENSION: usize = 512;
const PIXELS: usize = SIDE * SIDE;

pub fn compare_embeddings(
    a_space: &str,
    a: &[f32],
    b_space: &str,
    b: &[f32],
) -> Result<f64, EngineError> {
    if a_space.is_empty() || a_space != b_space {
        return Err(EngineError::incompatible_embedding(
            "Expected embeddings from the same space",
        ));
    }
    if a.len() != DIMENSION || b.len() != DIMENSION {
        return Err(EngineError::incompatible_embedding(
            "Expected two 512-dimensional embeddings",
        ));
    }

    let (mut dot, mut a_norm, mut b_norm) = (0.0, 0.0, 0.0);
    for (&a, &b) in a.iter().zip(b) {
        if !a.is_finite() || !b.is_finite() {
            return Err(EngineError::incompatible_embedding(
                "Embedding values must be finite",
            ));
        }
        let (a, b) = (f64::from(a), f64::from(b));
        dot += a * b;
        a_norm += a * a;
        b_norm += b * b;
    }
    if a_norm == 0.0 || b_norm == 0.0 {
        return Err(EngineError::incompatible_embedding(
            "Embedding norms must be non-zero",
        ));
    }
    Ok((dot / (a_norm.sqrt() * b_norm.sqrt())).clamp(-1.0, 1.0))
}

pub(crate) struct Recognizer {
    session: Session,
    tensor: Tensor<f32>,
    aligned: RgbImage,
    space_id: String,
}

impl Recognizer {
    pub(crate) fn open(model: &[u8], sha256: &str, threads: usize) -> Result<Self, EngineError> {
        let environment = crate::environment().map_err(EngineError::incompatible)?;
        let session = Session::builder(environment)
            .map_err(EngineError::incompatible)?
            .with_intra_threads(threads)
            .map_err(EngineError::incompatible)?
            .commit_from_memory(model)
            .map_err(EngineError::incompatible)?;
        let inputs = session.inputs();
        let outputs = session.outputs();
        if inputs.len() != 1
            || !compatible(inputs[0].dtype(), &[1, 3, SIDE as i64, SIDE as i64])
            || outputs.len() != 1
            || !compatible(outputs[0].dtype(), &[1, DIMENSION as i64])
        {
            return Err(EngineError::incompatible(
                "Expected an ArcFace RGB 112 × 112 input and 512-dimensional output",
            ));
        }

        let tensor = Tensor::from_array(([1, 3, SIDE, SIDE], vec![0_f32; PIXELS * 3]))
            .map_err(EngineError::incompatible)?;
        Ok(Self {
            session,
            tensor,
            aligned: RgbImage::new(SIDE as u32, SIDE as u32),
            space_id: format!("{ADAPTER}:sha256:{sha256}"),
        })
    }

    pub(crate) fn embed(
        &mut self,
        image: &ImageBuffer<Rgb<u8>, impl Deref<Target = [u8]>>,
        landmarks: [Point; 5],
    ) -> Result<Option<Embedding>, EngineError> {
        if !align::crop(image, landmarks, &mut self.aligned) {
            return Ok(None);
        }
        let (_, tensor) = self
            .tensor
            .try_extract_tensor_mut()
            .map_err(EngineError::incompatible)?;
        for (i, pixel) in self.aligned.pixels().enumerate() {
            for (channel, value) in pixel.0.into_iter().enumerate() {
                tensor[channel * PIXELS + i] = (f32::from(value) - 127.5) / 127.5;
            }
        }

        let outputs = self
            .session
            .run(ort::inputs![&self.tensor])
            .map_err(EngineError::incompatible)?;
        let (shape, values) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(EngineError::incompatible)?;
        if shape[..] != [1, DIMENSION as i64] {
            return Err(EngineError::incompatible(
                "Unexpected ArcFace output dimensions",
            ));
        }
        let norm = values
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(EngineError::incompatible(
                "ArcFace produced an invalid embedding norm",
            ));
        }
        let values = values
            .iter()
            .map(|&value| (f64::from(value) / norm) as f32)
            .collect();
        Ok(Some(Embedding {
            space_id: self.space_id.clone(),
            values,
        }))
    }
}

fn compatible(value: &ValueType, expected: &[i64]) -> bool {
    let ValueType::Tensor {
        ty: TensorElementType::Float32,
        shape,
        ..
    } = value
    else {
        return false;
    };
    shape.len() == expected.len() && matches!(shape[0], -1 | 1) && shape[1..] == expected[1..]
}
