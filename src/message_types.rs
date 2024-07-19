use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};

use crate::executor::Packet;

// use crate::executor::FullPacket;

#[pyclass]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct MessageServer {
    pub purpose: MessageType,
}

#[pymethods]
impl MessageServer {
    #[new]
    pub fn new(purpose: MessageType) -> Self {
        MessageServer { purpose }
    }
}

impl<'source> FromPyObject<'source> for MessageType {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        let value: MessageType = serde_json::from_value(py_any_to_value(obj)?)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("{}", e)))?;
        Ok(value)
    }
}

fn py_any_to_value(obj: &PyAny) -> PyResult<serde_json::Value> {
    let str_obj: &str = obj.extract()?;
    serde_json::from_str(str_obj).map_err(|e| PyErr::new::<PyValueError, _>(format!("{}", e)))
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum MessageType {
    Initialise(Entity),
    JobSendPath(String),
    StatisticsSend(Statistics),
    RequestingNet,
    NewNetworkPath(String),
    IdentityConfirmation((Entity, usize)),
    JobSendData(Vec<DataFileType>),
    NewNetworkData(Vec<u8>),
    TBLink((String, String)),
    CreateTB,
    RequestingTBLink,
    EvaluationRequest(ExternalPacket), // use Vec<f32> to handle raw input data
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct ExternalPacket {
    pub data: Vec<f32>,
    pub datagen_id: usize,
    pub mcts_id: usize,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum DataFileType {
    OffFile(Vec<u8>),
    MetaDataFile(Vec<u8>),
    BinFile(Vec<u8>),
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum Entity {
    RustDataGen,
    PythonTraining,
    TBHost,
    GUIMonitor,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum Statistics {
    NodesPerSecond(usize),
    EvalsPerSecond(usize),
}

#[pymodule]
#[pyo3(name = "tz_rust")]
fn tz_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MessageServer>()?;
    Ok(())
}
