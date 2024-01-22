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
