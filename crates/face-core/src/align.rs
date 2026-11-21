use std::ops::Deref;

use image::{ImageBuffer, Rgb, RgbImage};

use crate::Point;

pub(crate) const SIDE: usize = 112;
const TEMPLATE: [[f64; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

pub(crate) fn crop(
    image: &ImageBuffer<Rgb<u8>, impl Deref<Target = [u8]>>,
    landmarks: [Point; 5],
    output: &mut RgbImage,
) -> bool {
    let Some(transform) = inverse_similarity(landmarks) else {
        return false;
    };
    for (x, y, pixel) in output.enumerate_pixels_mut() {
        let x = f64::from(x);
        let y = f64::from(y);
        let [a, b, tx, ty] = transform;
        *pixel = sample(image, a * x - b * y + tx, b * x + a * y + ty);
    }
    true
}

fn inverse_similarity(landmarks: [Point; 5]) -> Option<[f64; 4]> {
    let points = landmarks.map(|p| [f64::from(p.x), f64::from(p.y)]);
    let [sx, sy] = centroid(&points);
    let [dx, dy] = centroid(&TEMPLATE);
    let (mut xx, mut yy, mut xy, mut dot, mut cross) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for ([x, y], [u, v]) in points.into_iter().zip(TEMPLATE) {
        let (x, y, u, v) = (x - sx, y - sy, u - dx, v - dy);
        xx += x * x;
        yy += y * y;
        xy += x * y;
        dot += x * u + y * v;
        cross += x * v - y * u;
    }
    let energy = xx + yy;
    if !energy.is_finite() || xx * yy - xy * xy <= 1e-10 * energy * energy {
        return None;
    }
    // Fit an orientation-preserving similarity from the image to the template, then invert it.
    let a = dot / energy;
    let b = cross / energy;
    let squared_scale = a * a + b * b;
    if !squared_scale.is_finite() || squared_scale <= f64::EPSILON {
        return None;
    }
    let (a, b) = (a / squared_scale, -b / squared_scale);
    Some([a, b, sx - a * dx + b * dy, sy - b * dx - a * dy])
}

fn centroid(points: &[[f64; 2]; 5]) -> [f64; 2] {
    std::array::from_fn(|axis| points.iter().map(|point| point[axis]).sum::<f64>() / 5.0)
}

fn sample(image: &ImageBuffer<Rgb<u8>, impl Deref<Target = [u8]>>, x: f64, y: f64) -> Rgb<u8> {
    // Use the same 1/32-pixel interpolation grid as OpenCV INTER_LINEAR.
    let x = (x * 32.0).round() / 32.0;
    let y = (y * 32.0).round() / 32.0;
    if x <= -1.0 || y <= -1.0 || x >= f64::from(image.width()) || y >= f64::from(image.height()) {
        return Rgb([0; 3]);
    }
    let (left, top) = (x.floor() as i64, y.floor() as i64);
    let (dx, dy) = (x - x.floor(), y - y.floor());
    let mut color = [0.0; 3];
    for (py, wy) in [(top, 1.0 - dy), (top + 1, dy)] {
        for (px, wx) in [(left, 1.0 - dx), (left + 1, dx)] {
            if px >= 0 && py >= 0 && px < i64::from(image.width()) && py < i64::from(image.height())
            {
                for (value, channel) in color
                    .iter_mut()
                    .zip(image.get_pixel(px as u32, py as u32).0)
                {
                    *value += f64::from(channel) * wx * wy;
                }
            }
        }
    }
    Rgb(color.map(|value| value.round() as u8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligns_a_known_rotation_scale_and_translation() {
        let original = RgbImage::from_fn(230, 230, |x, y| Rgb([x as u8, y as u8, 127]));
        let landmarks = TEMPLATE.map(|[x, y]| Point {
            x: (224.0 - 2.0 * y) as f32,
            y: (2.0 * x - 2.0) as f32,
        });
        let mut aligned = RgbImage::new(SIDE as u32, SIDE as u32);
        assert!(crop(&original, landmarks, &mut aligned));
        assert_eq!(aligned.get_pixel(20, 30).0, [164, 38, 127]);
        assert_eq!(aligned.get_pixel(0, 0).0, [0, 0, 0]);
        assert_eq!(aligned.get_pixel(111, 111).0, [2, 220, 127]);
    }

    #[test]
    fn matches_opencv_pixels_with_subpixel_and_noisy_landmarks() {
        let image = RgbImage::from_fn(180, 164, |x, y| {
            Rgb([(x + y) as u8, (x + 2 * y) as u8, (3 * x + y) as u8])
        });
        let noise = [
            [1.2, -0.2],
            [0.0, 0.7],
            [0.6, 0.0],
            [-1.1, 0.1],
            [0.0, -0.8],
        ];
        let landmarks = std::array::from_fn(|i| {
            let [x, y] = TEMPLATE[i].map(|coordinate| f64::from(coordinate as f32));
            Point {
                x: (1.2 * x - 0.36 * y + 22.0 + noise[i][0]) as f32,
                y: (0.36 * x + 1.2 * y - 3.0 + noise[i][1]) as f32,
            }
        });
        let mut aligned = RgbImage::new(SIDE as u32, SIDE as u32);
        assert!(crop(&image, landmarks, &mut aligned));
        // Reference: skimage SimilarityTransform followed by OpenCV warpAffine (INTER_LINEAR).
        let pixels = [
            (0, 0, [0, 0, 0]),
            (0, 111, [0, 0, 0]),
            (16, 7, [51, 63, 132]),
            (20, 30, [76, 116, 149]),
            (51, 68, [156, 253, 18]),
            (80, 63, [197, 43, 132]),
            (100, 100, [3, 156, 213]),
            (111, 111, [0, 0, 0]),
        ];
        for (x, y, expected) in pixels {
            for (value, reference) in aligned.get_pixel(x, y).0.into_iter().zip(expected) {
                assert!(
                    value.abs_diff(reference) <= 1,
                    "pixel ({x}, {y}): {value} vs {reference}"
                );
            }
        }
    }

    #[test]
    fn rejects_coincident_and_collinear_points() {
        assert!(inverse_similarity([Point { x: 10.0, y: 20.0 }; 5]).is_none());
        let line = std::array::from_fn(|i| Point {
            x: i as f32,
            y: 2.0 * i as f32,
        });
        assert!(inverse_similarity(line).is_none());
    }
}
