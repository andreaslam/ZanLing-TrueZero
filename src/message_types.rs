use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};

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

#[pyclass]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum MessageType {
    Initialise(Entity),
    JobSendPath(String),
    StatisticsSend(Statistics),
    RequestingNet(),
    NewNetworkPath(String),
    IdentityConfirmation((Entity, usize)),
    JobSendData(Vec<DataFileType>),
    NewNetworkData(Vec<u8>),
    TBLink((String, String)),
    CreateTB(),
    RequestingTBLink(),
    EvaluationRequest(ExternalPacket), // use Vec<f32> to handle raw input data
}
#[pyclass]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct ExternalPacket {
    pub data: Vec<f32>,
    pub datagen_id: usize,
    pub mcts_id: usize,
}

#[pyclass]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum DataFileType {
    OffFile(Vec<u8>),
    MetaDataFile(Vec<u8>),
    BinFile(Vec<u8>),
}

#[pyclass(eq, eq_int)]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Eq)]
pub enum Entity {
    RustDataGen,
    PythonTraining,
    TBHost,
    GUIMonitor,
}

#[pyclass]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum Statistics {
    NodesPerSecond(usize),
    EvalsPerSecond(usize),
}

#[pymodule]
#[pyo3(name = "tzrust")]
fn tzrust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MessageServer>()?;
    m.add_class::<ExternalPacket>()?;
    m.add_class::<MessageType>()?;
    m.add_class::<DataFileType>()?;
    m.add_class::<Statistics>()?;
    m.add_class::<Entity>()?;
    Ok(())
}
