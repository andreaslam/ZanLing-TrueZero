use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct MessageServer {
    pub purpose: MessageType,
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
    NodesPerSecond(f32),
    EvalsPerSecond(f32),
}
