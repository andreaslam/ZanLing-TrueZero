use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct ServerMessageSend {
    // generator to server
    pub is_continue: bool,                   // set false if sending stop signal
    pub initialise_identity: Option<String>, // rust-datagen or python-training
    pub nps: Option<f32>,                    // nps statistics
    pub evals_per_second: Option<f32>,       // evals/s statistics
    pub job_path: Option<String>,            // game file path
    pub net_path: Option<String>,            // carry the path of current net
    pub has_net: bool, // set true if python (since not applicable), but for rust, if still waiting for any net (NOT if there is a net but it's not updated), set to true
    pub purpose: String,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct ServerMessageRecv {
    // server to generator
    pub verification: Option<String>, // string to verify identity
    pub net_path: Option<String>,
}

// =============================================================================================================================================

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
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]

pub enum Entity {
    RustDataGen,
    PythonTraining,

    GUIMonitor,
}

// impl fmt::Display for MessageType {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         let msg: String;
//         match &self.nodes[0].mv {
//             Some(mv) => {
//                 let m1 = "This is object of type Node and represents action ";
//                 let m2 = format!("{}", mv);
//                 msg = m1.to_string() + &m2;
//             }
//             None => {
//                 msg = "Node at starting board position".to_string();
//             }
//         }
//         write!(f, "{}", msg)
//     }
// }

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum Statistics {
    NodesPerSecond(f32),
    EvalsPerSecond(f32),
}
