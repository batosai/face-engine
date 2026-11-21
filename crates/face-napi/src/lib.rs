use std::path::Path;

use napi_derive::napi;

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
