use crate::ActivationData;
use rand::distributions::{Distribution, Uniform};
use rand::RngCore;

pub struct Neuron {
    bias: f64,
    previous_weights: ActivationData,
}

impl Neuron {
    pub fn new(
        bias: f64,
        uniform: &Uniform<f64>,
        mut rng: Box<&mut dyn RngCore>,
        previous_neurons: usize,
    ) -> Self {
        let mut previous_weights = vec![];

        for _ in 0..previous_neurons {
            previous_weights.push(uniform.sample(&mut rng));
        }

        Self {
            bias,
            previous_weights,
        }
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }

    pub fn previous_weights(&self) -> &ActivationData {
        &self.previous_weights
    }

    pub fn activate(&self, previous_activation: &[f64]) -> f64 {
        let mut result = self.bias;

        for i in 0..self.previous_weights.len() {
            let activation = previous_activation
                .get(i)
                .expect("Unexpected error happened");
            let weight = self
                .previous_weights
                .get(i)
                .expect("Unexpected error happened");

            result += activation * weight;
        }

        result *= -1.0;

        1.0 / (1.0 + result.exp())
    }

    pub fn adjust_previous_weights(&mut self, offset: f64) {
        for weight in self.previous_weights.iter_mut() {
            *weight += offset * *weight;
        }
    }
}
