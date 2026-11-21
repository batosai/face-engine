use std::{
    fs::{self, File},
    io::{Cursor, Read},
    path::Path,
};

use image::{DynamicImage, ImageDecoder, ImageError, ImageReader, Limits, RgbImage};

use crate::EngineError;

const MAX_ENCODED_BYTES: usize = 32 * 1024 * 1024;
const MAX_PIXELS: u64 = 32_000_000;
const MAX_SIDE: u32 = 16_384;

pub(crate) fn read_file(path: &Path) -> Result<Vec<u8>, EngineError> {
    if !fs::metadata(path)
        .map_err(EngineError::invalid_image)?
        .is_file()
    {
        return Err(EngineError::invalid_image("Expected a regular image file"));
    }
    let file = File::open(path).map_err(EngineError::invalid_image)?;
    let metadata = file.metadata().map_err(EngineError::invalid_image)?;
    if !metadata.is_file() {
        return Err(EngineError::invalid_image("Expected a regular image file"));
    }
    if metadata.len() > MAX_ENCODED_BYTES as u64 {
        return Err(EngineError::limit("Encoded image exceeds 32 MiB"));
    }
    let mut encoded = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_ENCODED_BYTES as u64 + 1)
        .read_to_end(&mut encoded)
        .map_err(EngineError::invalid_image)?;
    Ok(encoded)
}

pub(crate) fn validate_dimensions(width: u32, height: u32) -> Result<(), EngineError> {
    if width == 0 || height == 0 {
        return Err(EngineError::invalid_image(
            "Image dimensions must be positive",
        ));
    }
    if width > MAX_SIDE || height > MAX_SIDE || u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(EngineError::limit(
            "Image exceeds the dimension or pixel limit",
        ));
    }
    Ok(())
}

pub(crate) fn decode(encoded: &[u8]) -> Result<RgbImage, EngineError> {
    if encoded.len() > MAX_ENCODED_BYTES {
        return Err(EngineError::limit("Encoded image exceeds 32 MiB"));
    }
    let mut reader = ImageReader::new(Cursor::new(encoded))
        .with_guessed_format()
        .map_err(EngineError::invalid_image)?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_SIDE);
    limits.max_image_height = Some(MAX_SIDE);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let mut decoder = reader.into_decoder().map_err(decode_error)?;
    let (width, height) = decoder.dimensions();
    validate_dimensions(width, height)?;
    let orientation = decoder.orientation().map_err(decode_error)?;
    let mut image = DynamicImage::from_decoder(decoder).map_err(decode_error)?;
    image.apply_orientation(orientation);
    Ok(image.into_rgb8())
}

fn decode_error(error: ImageError) -> EngineError {
    match error {
        ImageError::Limits(_) => EngineError::limit("Image exceeds decoder limits"),
        _ => EngineError::invalid_image(error),
    }
}
