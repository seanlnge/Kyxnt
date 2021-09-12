use crate::kyxnt::networks::functions::learning_rate::LearningRate;

use std::process;
use rand::prelude::*;

pub struct Standard {
    pub bias: f64,
    pub weights: Vec<f64>,

    pub activation: f64,
    pub change_summations: (f64, Vec<f64>, usize),
    pub input_indexes: Vec<usize>,

    pub previous_weight_gradients: Vec<Vec<f64>>, // PreviousIterations< Weights<f64> >
    pub previous_bias_gradients: Vec<f64>,

    pub activation_function: Box<dyn Fn(f64) -> f64>, // Fn(total) -> activation
    pub derivative_function: Box<dyn Fn(f64) -> f64>  // Fn(activation) -> derivative
}
impl Standard {
    pub fn new(input_indexes: Vec<usize>, activation: &str) -> Self {
        let mut rng = thread_rng();
        let [activation_function, derivative_function]: [Box<dyn Fn(f64) -> f64>; 2] = match activation {
            "sigmoid"|"softplus"|"logistic" => [
                Box::new(|x| 1.0 / (1.0 + (-x).exp())),
                Box::new(|o| o * (1.0 - o))
            ],
            "tanh" => [
                Box::new(|x| (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())),
                Box::new(|o| 1.0 - o.powi(2))
            ],
            "relu" => [
                Box::new(|x| x.max(0.0)),
                Box::new(|o| o.signum())
            ],
            "none" => [
                Box::new(|x| x),
                Box::new(|_| 1.0)
            ],
            _ => {
                println!("Unknown activation function '{}'", activation);
                process::exit(1);
            }
        };

        Self {
            bias: rng.gen_range(-1.0..1.0),
            weights: input_indexes.iter().map(|_| rng.gen_range(-1.0..1.0)).collect(),

            activation: 0.0,
            change_summations: (0.0, input_indexes.iter().map(|_| 0.0).collect(), 0),

            previous_weight_gradients: input_indexes.iter().map(|_| vec![]).collect(),
            previous_bias_gradients: vec![],

            input_indexes,

            activation_function,
            derivative_function
        }
    }

    // Returns node activation
    pub fn evaluate(&mut self, inputs: &Vec<f64>) -> f64 {
        let mut total = self.bias;

        for (index, value) in inputs.iter().enumerate() {
            total += value * self.weights[index];
        }

        self.activation = self.activation_function.as_ref()(total);
        self.activation
    }

    // Returns (Requested Inputs, [Weights, Bias])
    pub fn backpropagate(&mut self, input_nodes: &Vec<f64>, requested_activation: f64) -> Vec<f64> {
        let mut optimal_inputs = vec![];

        // Calculate d(cost) wrt pre_activation value
        let dadz = self.derivative_function.as_ref()(self.activation);
        let dcdz = 2.0 * (self.activation - requested_activation) * dadz;
        
        // d(pre_activation value) wrt bias
        self.change_summations.0 += dcdz;

        // Calculate d(pre_activation value) wrt each variable
        for (index, input) in input_nodes.iter().enumerate() {
            optimal_inputs.push(dcdz * self.weights[index]);
            self.change_summations.1[index] += dcdz * input;
        }

        // Increment iterations and return
        self.change_summations.2 += 1;
        optimal_inputs
    }

    pub fn apply_changes(&mut self, learning_rate_function: &LearningRate, parameters: &Vec<f64>) {
        for (i, weight_change) in self.change_summations.1.iter().enumerate() {
            let gradient = weight_change / self.change_summations.2 as f64;
            let adjusted_gradient = match learning_rate_function {
                LearningRate::SingleParameter(ref func) => func(gradient),
                LearningRate::Decay(ref func) => func(gradient, parameters[0]),
                LearningRate::Momentum(ref func) => {
                    let pgi = self.previous_bias_gradients.len() as isize - 1;
                    let previous_gradient = if pgi >= 0 { self.previous_weight_gradients[i][pgi as usize] } else { 0.0 };
                    func(gradient, previous_gradient)
                }
            };

            self.previous_weight_gradients[i].push(-adjusted_gradient);
            self.weights[i] -= adjusted_gradient;
        }

        let gradient = self.change_summations.0 / self.change_summations.2 as f64;
        let adjusted_gradient = match learning_rate_function {
            LearningRate::SingleParameter(ref func) => func(gradient),
            LearningRate::Decay(ref func) => func(gradient, parameters[0]),
            LearningRate::Momentum(ref func) => {
                let pgi = self.previous_bias_gradients.len() as isize - 1;
                let previous_gradient = if pgi >= 0 { self.previous_bias_gradients[pgi as usize] } else { 0.0 };
                func(gradient, previous_gradient)
            }
        };

        self.previous_bias_gradients.push(-adjusted_gradient);
        self.bias -= adjusted_gradient;
    }
}