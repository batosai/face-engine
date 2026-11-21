use std::ops::Deref;

use image::{ImageBuffer, Rgb};
use ort::{
    session::{Session, SessionOutputs},
    value::{Tensor, TensorElementType, ValueType},
};

use crate::{Detection, EngineError, Point, Rect};

const SIDE: usize = 640;
const PLANES: usize = SIDE * SIDE;
const LEVELS: [(usize, [&str; 3]); 3] = [
    (8, ["443", "446", "449"]),
    (16, ["468", "471", "474"]),
    (32, ["493", "496", "499"]),
];
const OUTPUT_WIDTHS: [usize; 3] = [1, 4, 10];
const SCORE_MIN: f32 = 0.5;
const NMS_THRESHOLD: f32 = 0.4;
const MAX_FACES: usize = 256;

pub(crate) struct Detector {
    session: Session,
    tensor: Tensor<f32>,
}

impl Detector {
    pub(crate) fn open(model: &[u8], threads: usize) -> Result<Self, EngineError> {
        let env = crate::environment().map_err(EngineError::incompatible)?;
        let session = Session::builder(env)
            .map_err(EngineError::incompatible)?
            .with_intra_threads(threads)
            .map_err(EngineError::incompatible)?
            .commit_from_memory(model)
            .map_err(EngineError::incompatible)?;

        let inputs = session.inputs();
        if inputs.len() != 1
            || inputs[0].name() != "input.1"
            || !tensor_shape(inputs[0].dtype(), &[1, 3, SIDE as i64, SIDE as i64])
        {
            return Err(EngineError::incompatible("Unexpected SCRFD input"));
        }
        if session.outputs().len() != 9 {
            return Err(EngineError::incompatible("Expected nine SCRFD outputs"));
        }
        for (stride, names) in LEVELS {
            for (name, columns) in names.into_iter().zip(OUTPUT_WIDTHS) {
                let outlet = session
                    .outputs()
                    .iter()
                    .find(|output| output.name() == name)
                    .ok_or_else(|| {
                        EngineError::incompatible(format!("Missing SCRFD output {name}"))
                    })?;
                if !tensor_shape(outlet.dtype(), &[rows(stride) as i64, columns as i64]) {
                    return Err(EngineError::incompatible(format!(
                        "Unexpected SCRFD output shape for {name}"
                    )));
                }
            }
        }

        let tensor = Tensor::from_array(([1, 3, SIDE, SIDE], vec![0_f32; PLANES * 3]))
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
        let scale = prepare(image, pixels);
        let outputs = self
            .session
            .run(ort::inputs![&self.tensor])
            .map_err(EngineError::incompatible)?;
        let mut candidates = Vec::new();
        for (stride, names) in LEVELS {
            let [scores, boxes, points] = read_level(&outputs, stride, names)?;
            let grid = SIDE / stride;
            let stride = stride as f32;
            for (row, score) in scores.iter().copied().enumerate() {
                if !(SCORE_MIN..=1.0).contains(&score) {
                    continue;
                }
                let cell = row / 2;
                let x = (cell % grid) as f32 * stride;
                let y = (cell / grid) as f32 * stride;
                let b = &boxes[row * 4..][..4];
                let p = &points[row * 10..][..10];
                if b.iter().chain(p).any(|value| !value.is_finite()) {
                    continue;
                }
                let edges = [
                    (x - b[0] * stride) / scale,
                    (y - b[1] * stride) / scale,
                    (x + b[2] * stride) / scale,
                    (y + b[3] * stride) / scale,
                ];
                let landmarks = std::array::from_fn(|i| Point {
                    x: (x + p[i * 2] * stride) / scale,
                    y: (y + p[i * 2 + 1] * stride) / scale,
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

fn rows(stride: usize) -> usize {
    (SIDE / stride).pow(2) * 2
}

fn read_level<'a>(
    outputs: &'a SessionOutputs,
    stride: usize,
    names: [&str; 3],
) -> Result<[&'a [f32]; 3], EngineError> {
    let mut tensors = [&[][..]; 3];
    for (index, (name, columns)) in names.into_iter().zip(OUTPUT_WIDTHS).enumerate() {
        let output = outputs
            .get(name)
            .ok_or_else(|| EngineError::incompatible(format!("Missing SCRFD output {name}")))?;
        let (shape, values) = output
            .try_extract_tensor::<f32>()
            .map_err(EngineError::incompatible)?;
        if shape[..] != [rows(stride) as i64, columns as i64] {
            return Err(EngineError::incompatible(format!(
                "Unexpected SCRFD output dimensions for {name}"
            )));
        }
        tensors[index] = values;
    }
    Ok(tensors)
}

fn prepare(image: &ImageBuffer<Rgb<u8>, impl Deref<Target = [u8]>>, pixels: &mut [f32]) -> f32 {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let longest = width.max(height);
    let w = (width * SIDE / longest).max(1);
    let h = (height * SIDE / longest).max(1);
    let xs: Vec<_> = (0..w).map(|x| source_coordinate(x, w, width)).collect();
    pixels.fill(-127.5 / 128.0);

    for y in 0..h {
        let (top, bottom, dy) = source_coordinate(y, h, height);
        for (x, &(left, right, dx)) in xs.iter().enumerate() {
            let a = image.get_pixel(left, top).0;
            let b = image.get_pixel(right, top).0;
            let c = image.get_pixel(left, bottom).0;
            let d = image.get_pixel(right, bottom).0;
            for channel in 0..3 {
                let upper = f32::from(a[channel]) * (1.0 - dx) + f32::from(b[channel]) * dx;
                let lower = f32::from(c[channel]) * (1.0 - dx) + f32::from(d[channel]) * dx;
                let rgb = (upper * (1.0 - dy) + lower * dy).round();
                pixels[channel * PLANES + y * SIDE + x] = (rgb - 127.5) / 128.0;
            }
        }
    }
    h as f32 / height as f32
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
