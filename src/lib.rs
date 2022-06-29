pub mod model;
pub mod neuron;

pub type ActivationData = Vec<f64>;

pub struct TrainingData {
    data: ActivationData,
    expected: ActivationData,
}

impl TrainingData {
    pub fn new(data: ActivationData, expected: ActivationData) -> Self {
        Self { data, expected }
    }
}
