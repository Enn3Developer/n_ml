use crate::neuron::Neuron;
use crate::TrainingData;
use rand::distributions::Uniform;
use rand::thread_rng;

pub struct Model {
    layers: Vec<Vec<Neuron>>,
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

impl Model {
    pub fn new() -> Self {
        Self { layers: vec![] }
    }

    pub fn add_layer(&mut self, neurons: u64) {
        let uniform = Uniform::new_inclusive(0.0, 1.0);
        let mut rng = thread_rng();
        let previous_neurons = if self.layers.is_empty() {
            0
        } else {
            self.layers.last().unwrap().len()
        };

        let mut new_layer = vec![];

        for _ in 0..neurons {
            new_layer.push(Neuron::new(
                -10.0,
                &uniform,
                Box::new(&mut rng),
                previous_neurons,
            ))
        }
    }

    pub fn evaluate(&self, data: &Vec<f64>) -> Vec<f64> {
        assert!(
            self.layers.len() >= 2,
            "Not a valid model, a valid model should have at least 2 layers",
        );
        assert_eq!(
            data.len(),
            self.layers.first().unwrap().len(),
            "The data provided don't have the expected len: {} vs the model {}",
            data.len(),
            self.layers.first().unwrap().len(),
        );

        let mut previous_activations = data.clone();

        for i in 1..self.layers.len() {
            let neurons = self.layers.get(i).unwrap();
            let mut current_activations = vec![];

            for neuron in neurons {
                current_activations.push(neuron.activate(&previous_activations));
            }

            previous_activations = current_activations;
        }

        previous_activations
    }

    pub fn train(&self, training_data: Vec<TrainingData>, batch: usize, epoch: u64) {
        assert!(batch > 0, "The batch size cannot be 0");
        assert!(epoch > 0, "The epoch amount cannot be 0");
        assert!(
            !training_data.is_empty(),
            "Training data should have data inside"
        );
        assert!(
            self.layers.len() >= 2,
            "Not a valid model, a valid model should have at least 2 layers",
        );
        assert_eq!(
            training_data.first().unwrap().expected.len(),
            self.layers.last().unwrap().len(),
            "The expected results should be the same length as the last layer: {} vs the model {}",
            training_data.first().unwrap().expected.len(),
            self.layers.last().unwrap().len(),
        );

        let mut batch_results = vec![];

        for _ in 0..epoch {
            for data in &training_data {
                let evaluation_results = self.evaluate(&data.data);
                let divergences: Vec<f64> = evaluation_results
                    .iter()
                    .zip(&data.expected)
                    .map(|(value, expected)| value - expected)
                    .collect();

                batch_results.push(divergences);

                if batch_results.len() >= batch {
                    let batch_results_len = batch_results.first().unwrap().len();
                    let mut offset_adjustments = vec![];

                    for i in 0..batch_results_len {
                        let mut sum = 0.0;

                        for divergences in &batch_results {
                            sum += divergences.get(i).unwrap();
                        }

                        offset_adjustments.push(sum / batch_results_len as f64);
                    }

                    // TODO: make backpropagation here
                }
            }
        }
    }
}
