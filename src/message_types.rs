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
