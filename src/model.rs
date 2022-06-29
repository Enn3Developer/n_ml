use crate::neuron::Neuron;
use crate::{ActivationData, TrainingData};
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

    pub fn evaluate(&self, data: &ActivationData) -> ActivationData {
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

    pub fn train(&mut self, training_data: Vec<TrainingData>, batch: usize, epoch: u64) {
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
                let divergences: ActivationData = evaluation_results
                    .iter()
                    .zip(&data.expected)
                    .map(|(value, expected)| value - expected)
                    .collect();

                batch_results.push(divergences);

                if batch_results.len() >= batch {
                    self.fit(&mut batch_results);
                }
            }
        }
    }

    fn fit(&mut self, batch_results: &mut Vec<ActivationData>) {
        let batch_results_len = batch_results.first().unwrap().len();
        let mut offset_adjustments = vec![];

        for i in 0..batch_results_len {
            let mut sum = 0.0;

            for divergences in &mut *batch_results {
                sum += divergences.get(i).unwrap();
            }

            offset_adjustments.push(sum / batch_results_len as f64);
        }

        // backpropagation
        let mut i = self.layers.len();
        while i > 1 {
            i -= 1;
            let current_layer = self.layers.get_mut(i).unwrap();

            let previous_weights: Vec<ActivationData> = current_layer
                .iter()
                .map(|neuron| neuron.previous_weights().clone())
                .collect();
            let mut new_weights = vec![];

            // adjust the weights and get the newer ones
            for j in 0..offset_adjustments.len() {
                let offset_adjustment = *offset_adjustments.get(j).unwrap();
                let neuron = current_layer.get_mut(j).unwrap();
                neuron.adjust_previous_weights(offset_adjustment);
                new_weights.push(neuron.previous_weights().clone());
            }

            offset_adjustments.clear();

            // re-populate `offset_adjustments`
            let weights_len = new_weights.first().unwrap().len();
            for j in 0..weights_len {
                let mut sum = 0.0;

                for neuron_index in 0..new_weights.len() {
                    sum += previous_weights.get(neuron_index).unwrap().get(j).unwrap()
                        - new_weights.get(neuron_index).unwrap().get(j).unwrap()
                }

                offset_adjustments.push(sum / weights_len as f64);
            }
        }
    }
}
