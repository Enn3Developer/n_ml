pub mod model;
pub mod neuron;

pub struct TrainingData {
    data: Vec<f64>,
    expected: Vec<f64>,
}

impl TrainingData {
    pub fn new(data: Vec<f64>, expected: Vec<f64>) -> Self {
        Self { data, expected }
    }
}
