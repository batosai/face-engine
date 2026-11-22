use std::{path::Path, sync::OnceLock};

use ort::{environment::Environment, logging::LogLevel, session::Session, value::Outlet};

mod align;
mod embedding;
mod engine;
mod picture;
mod yunet;

pub use embedding::compare_embeddings;
pub use engine::{Detection, Embedding, Engine, EngineError, Face, ImageInput, Point, Rect};

fn environment() -> ort::Result<&'static Environment> {
    static ENVIRONMENT: OnceLock<Result<Environment, String>> = OnceLock::new();
    ENVIRONMENT
        .get_or_init(|| {
            ort::init()
                .with_telemetry(false)
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|message| ort::Error::new(message.clone()))
}

pub struct GraphInfo {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

pub struct Probe {
    pub runtime: &'static str,
    pub detector: GraphInfo,
    pub recognizer: GraphInfo,
}

pub fn probe_models(detector_path: &Path, recognizer_path: &Path) -> ort::Result<Probe> {
    let environment = environment()?;
    let detector = Session::builder(environment)?
        .with_intra_threads(1)?
        .commit_from_file(detector_path)?;
    let recognizer = Session::builder(environment)?
        .with_intra_threads(1)?
        .with_log_level(LogLevel::Error)?
        .commit_from_file(recognizer_path)?;

    Ok(Probe {
        runtime: ort::info(),
        detector: graph_info(&detector),
        recognizer: graph_info(&recognizer),
    })
}

fn graph_info(session: &Session) -> GraphInfo {
    GraphInfo {
        inputs: session.inputs().iter().map(describe_outlet).collect(),
        outputs: session.outputs().iter().map(describe_outlet).collect(),
    }
}

fn describe_outlet(outlet: &Outlet) -> String {
    format!("{}: {}", outlet.name(), outlet.dtype())
}
