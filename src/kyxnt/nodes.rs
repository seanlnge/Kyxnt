use crate::kyxnt::networks::functions::learning_rate;

use std::process;

mod standard;
mod input;

pub use standard::Standard;
pub use input::Input;

pub enum Node {
    Standard(Standard),
    Input(Input)
}

impl Node {
    pub fn activation(&self) -> f64 {
        match self {
            Node::Input(x) => x.activation,
            Node::Standard(x) => x.activation
        }
    }

    pub fn set(&mut self, value: f64) {
        match self {
            Node::Input(x) => x.activation = value,
            Node::Standard(x) => x.activation = value
        }
    } 

    pub fn evaluate(&mut self, inputs: &Vec<f64>) -> f64 {
        match self {
            Node::Standard(x) => x.evaluate(inputs),
            _ => process::exit(1)
        }
    }

    pub fn backpropagate(&mut self, inputs: &Vec<f64>, optimal_activation: f64) -> Vec<f64> {
        match self {
            Node::Standard(x) => x.backpropagate(inputs, optimal_activation),
            _ => process::exit(1)
        }
    }

    pub fn apply_changes(&mut self, learning_rate: &learning_rate::LearningRate) {
        match self {
            Node::Standard(x) => x.apply_changes(learning_rate),
            _ => process::exit(1)
        }
    }

    pub fn standard(&mut self) -> &mut Standard {
        match self {
            Node::Standard(x) => x,
            _ => process::exit(1)
        }
    }
}