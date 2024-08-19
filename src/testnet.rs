use tch::{CModule, Device};
pub struct Net {
    pub net: CModule,
    pub device: Device,
}

impl Net {
    /// creates a new `Net` instance by loading a model from the specified path
    pub fn new(path: &str) -> Self {
        let device = Device::Cuda(0);

        let mut net = tch::CModule::load_on_device(path, device).expect("ERROR");
        net.set_eval();

        Self {
            net: net,
            device: device,
        }
    }
}
