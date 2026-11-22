use std::ops::Deref;

use image::{ImageBuffer, Rgb};
use ort::{
    session::{Session, SessionOutputs},
    value::{Tensor, TensorElementType, ValueType},
};

use crate::{Detection, EngineError, Point, Rect};

const SIDE: usize = 640;
const PLANE: usize = SIDE * SIDE;
const SCORE_MIN: f32 = 0.6;
const NMS_THRESHOLD: f32 = 0.3;
const STRIDES: [usize; 3] = [8, 16, 32];
const OUTPUTS: [(&str, usize); 4] = [("cls", 1), ("obj", 1), ("bbox", 4), ("kps", 10)];

pub(crate) struct Detector {
    session: Session,
    tensor: Tensor<f32>,
}

impl Detector {
    pub(crate) fn open(adapter: &str, model: &[u8], threads: usize) -> Result<Self, EngineError> {
        if adapter != "yunet-640-v1" {
            return Err(EngineError::incompatible("Unsupported detector adapter"));
        }
        let session = Session::builder(crate::environment().map_err(EngineError::incompatible)?)
            .map_err(EngineError::incompatible)?
            .with_intra_threads(threads)
            .map_err(EngineError::incompatible)?
            .commit_from_memory(model)
            .map_err(EngineError::incompatible)?;
        let inputs = session.inputs();
        if inputs.len() != 1
            || inputs[0].name() != "input"
            || !tensor_shape(inputs[0].dtype(), &[1, 3, SIDE as i64, SIDE as i64])
        {
            return Err(EngineError::incompatible("Unexpected YuNet input"));
        }
        for stride in STRIDES {
            for (prefix, columns) in OUTPUTS {
                let name = format!("{prefix}_{stride}");
                let expected = [1, (SIDE / stride).pow(2) as i64, columns as i64];
                if !session
                    .outputs()
                    .iter()
                    .any(|output| output.name() == name && tensor_shape(output.dtype(), &expected))
                {
                    return Err(EngineError::incompatible(format!(
                        "Missing or incompatible YuNet output {name}"
                    )));
                }
            }
        }
        let tensor = Tensor::from_array(([1, 3, SIDE, SIDE], vec![0.0; 3 * PLANE]))
            .map_err(EngineError::incompatible)?;
        Ok(Self { session, tensor })
    }

    pub(crate) fn detect(
        &mut self,
        image: &ImageBuffer<Rgb<u8>, impl Deref<Target = [u8]>>,
    ) -> Result<Vec<Detection>, EngineError> {
        let (_, pixels) = self
            .tensor
            .try_extract_tensor_mut()
            .map_err(EngineError::incompatible)?;
        let scale = SIDE as f32 / image.width().max(image.height()) as f32;
        let width = (image.width() as f32 * scale)
            .round()
            .clamp(1.0, SIDE as f32) as usize;
        let height = (image.height() as f32 * scale)
            .round()
            .clamp(1.0, SIDE as f32) as usize;
        prepare(image, pixels, width, height);
        let outputs = self
            .session
            .run(ort::inputs![&self.tensor])
            .map_err(EngineError::incompatible)?;

        let mut candidates = Vec::new();
        for stride in STRIDES {
            let [classes, objects, boxes, keypoints] = read_level(&outputs, stride)?;
            let grid = SIDE / stride;
            let step = stride as f32;
            for (index, (&class, &object)) in classes.iter().zip(objects).enumerate() {
                let score = (class.clamp(0.0, 1.0) * object.clamp(0.0, 1.0)).sqrt();
                if !(SCORE_MIN..=1.0).contains(&score) {
                    continue;
                }
                let bounds = &boxes[index * 4..][..4];
                let column = (index % grid) as f32;
                let row = (index / grid) as f32;
                let center_x = (column + bounds[0]) * step;
                let center_y = (row + bounds[1]) * step;
                if !(0.0..width as f32).contains(&center_x)
                    || !(0.0..height as f32).contains(&center_y)
                {
                    continue;
                }
                let (half_w, half_h) = (bounds[2].exp() * step * 0.5, bounds[3].exp() * step * 0.5);
                let points = &keypoints[index * 10..][..10];
                if !half_w.is_finite()
                    || !half_h.is_finite()
                    || points.iter().any(|value| !value.is_finite())
                {
                    continue;
                }
                let edges = [
                    (center_x - half_w) / scale,
                    (center_y - half_h) / scale,
                    (center_x + half_w) / scale,
                    (center_y + half_h) / scale,
                ];
                let landmarks = std::array::from_fn(|i| Point {
                    x: (column + points[2 * i]) * step / scale,
                    y: (row + points[2 * i + 1]) * step / scale,
                });
                candidates.push(Candidate {
                    edges,
                    score,
                    landmarks,
                });
            }
        }
        Ok(suppress(
            candidates,
            image.width() as f32,
            image.height() as f32,
        ))
    }
}

fn read_level<'a>(
    outputs: &'a SessionOutputs<'_>,
    stride: usize,
) -> Result<[&'a [f32]; 4], EngineError> {
    let mut tensors = [&[][..]; 4];
    for (index, (prefix, columns)) in OUTPUTS.into_iter().enumerate() {
        let name = format!("{prefix}_{stride}");
        let output = outputs
            .get(&name)
            .ok_or_else(|| EngineError::incompatible(format!("Missing YuNet output {name}")))?;
        let (shape, values) = output
            .try_extract_tensor::<f32>()
            .map_err(EngineError::incompatible)?;
        if shape[..] != [1, (SIDE / stride).pow(2) as i64, columns as i64] {
            return Err(EngineError::incompatible(format!(
                "Unexpected YuNet output dimensions for {name}"
            )));
        }
        tensors[index] = values;
    }
    Ok(tensors)
}

fn prepare(
    image: &ImageBuffer<Rgb<u8>, impl Deref<Target = [u8]>>,
    pixels: &mut [f32],
    width: usize,
    height: usize,
) {
    let columns: Vec<_> = (0..width)
        .map(|x| source_coordinate(x, width, image.width() as usize))
        .collect();
    pixels.fill(0.0);
    for y in 0..height {
        let (top, bottom, dy) = source_coordinate(y, height, image.height() as usize);
        for (x, &(left, right, dx)) in columns.iter().enumerate() {
            let a = image.get_pixel(left, top).0;
            let b = image.get_pixel(right, top).0;
            let c = image.get_pixel(left, bottom).0;
            let d = image.get_pixel(right, bottom).0;
            for channel in 0..3 {
                let upper = f32::from(a[channel]) * (1.0 - dx) + f32::from(b[channel]) * dx;
                let lower = f32::from(c[channel]) * (1.0 - dx) + f32::from(d[channel]) * dx;
                pixels[(2 - channel) * PLANE + y * SIDE + x] =
                    (upper * (1.0 - dy) + lower * dy).round();
            }
        }
    }
}

const MAX_FACES: usize = 256;

fn tensor_shape(value: &ValueType, expected: &[i64]) -> bool {
    let ValueType::Tensor {
        ty: TensorElementType::Float32,
        shape,
        ..
    } = value
    else {
        return false;
    };
    shape.len() == expected.len()
        && shape
            .iter()
            .zip(expected)
            .all(|(&actual, &expected)| actual == -1 || actual == expected)
}

fn source_coordinate(index: usize, resized: usize, original: usize) -> (u32, u32, f32) {
    let coordinate = ((index as f64 + 0.5) * original as f64 / resized as f64 - 0.5)
        .clamp(0.0, (original - 1) as f64);
    let lower = coordinate.floor() as usize;
    (
        lower as u32,
        (lower + 1).min(original - 1) as u32,
        (coordinate - lower as f64) as f32,
    )
}

struct Candidate {
    edges: [f32; 4],
    score: f32,
    landmarks: [Point; 5],
}

fn suppress(mut candidates: Vec<Candidate>, width: f32, height: f32) -> Vec<Detection> {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept = Vec::new();
    let mut detections = Vec::new();
    for candidate in candidates {
        let [left, top, right, bottom] = candidate.edges;
        let (x, y) = (left.clamp(0.0, width), top.clamp(0.0, height));
        let w = right.clamp(0.0, width) - x;
        let h = bottom.clamp(0.0, height) - y;
        if w <= 0.0
            || h <= 0.0
            || kept
                .iter()
                .any(|edges| overlap(edges, &candidate.edges) > NMS_THRESHOLD)
        {
            continue;
        }
        kept.push(candidate.edges);
        detections.push(Detection {
            bounds: Rect {
                x,
                y,
                width: w,
                height: h,
            },
            score: candidate.score,
            landmarks: candidate.landmarks,
        });
        if detections.len() == MAX_FACES {
            break;
        }
    }
    detections
}

fn overlap(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let area = |r: &[f32; 4]| (r[2] - r[0] + 1.0) * (r[3] - r[1] + 1.0);
    let x = (a[2].min(b[2]) - a[0].max(b[0]) + 1.0).max(0.0);
    let y = (a[3].min(b[3]) - a[1].max(b[1]) + 1.0).max(0.0);
    let intersection = x * y;
    intersection / (area(a) + area(b) - intersection)
}
