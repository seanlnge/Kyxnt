use crate::kyxnt::nodes::*;

use rand::prelude::*;
use rand::seq::SliceRandom;

pub struct FeedForward {
    /*
    `nodes` contains list of all nodes contained in network organized by layer:
        Input nodes are at beginning
        Hidden nodes somewhere in middle
        Output nodes at end
    */
    pub nodes: Vec<Node>,

    /*
    `layers` contains organization of layers, stores where each layer in `nodes` ends:
        [2, 5, 6] stores a 2 - 3 - 1 network
        (layer[0]..layers[1]) = [2, 3, 4] - indexes of hidden layer `nodes`
        (0..layer[0]) = [0, 1] - indexes of input layer in `nodes`
    */
    pub layers: Vec<usize>,


    pub epochs_per_print: usize,
    pub learning_rate: f64,
    pub stochastic_noise: f64
}
impl FeedForward {
    pub fn new(input_size: usize) -> Self {
        Self {
            nodes: (0..input_size).map(|_| Node::Input(Input::new())).collect(),
            layers: vec![input_size],
            epochs_per_print: 0,
            stochastic_noise: 0.0,
            learning_rate: 0.1
        }
    }

    // Get vector of nodes corresponding to layer in FF model
    pub fn get_layer(&self, layer: usize) -> Vec<&Node> {
        let start = if layer == 0 { 0 } else { self.layers[layer-1] };
        let end = self.layers[layer];

        (start..end).map(|x| self.nodes.get(x).unwrap()).collect()
    }

    pub fn add_layer(&mut self, size: usize, activation: &str) {
        let start = if self.layers.len() == 1 { 0 } else { self.layers[self.layers.len() - 2] };
        let inputs: Vec<usize> = (start..*self.layers.last().unwrap()).collect();

        // Create `size` nodes
        for _ in 0..size {
            let node = Node::Standard(Standard::new(inputs.clone(), activation));
            self.nodes.push(node);
        }

        // Push end of inputs
        self.layers.push(self.nodes.len());
    }

    pub fn test(&mut self, input: &Vec<f64>) -> Vec<f64> {
        // Set input values for network
        for i in 0..self.layers[0] { self.nodes[i].set(input[i]) };

        // Loop over hidden/output layers and evaluate by feeding forward
        for layer in 1..self.layers.len() {
            // Find input values
            let input = self.get_layer(layer-1).iter().map(|x| x.activation()).collect();

            // Evaluate and change activations of each node in layer
            for i in self.layers[layer-1]..self.layers[layer] {
                self.nodes[i].evaluate(&input);
            }
        }

        // Return activations of last layer
        self.get_layer(self.layers.len()-1).iter().map(|x| x.activation()).collect()
    }

    pub fn stochastic_gradient_descent(&mut self, training_data: &Vec<[Vec<f64>; 2]>, epochs: usize) {
        let mut shuffle_rng = thread_rng();
        let mut noise_rng = thread_rng();

        let stochastic_noise = self.stochastic_noise;

        /*
            Constant is so that noise will not bias scale
            1 - 0.06054877864 is < 1 solution for x * (x - 1/8) = 1

            Because randomness has a linear gradient, iterations of
            scaling by noise will average to 1, since `min = 1/max`

            1/8 is arbitrary value
        */
        let mut noise = |epoch: usize| {
            let decay = 1.0 - epoch as f64 / epochs as f64;
            let random = noise_rng.gen_range(-1.0..1.0) / 8.0 - 0.06054877864;
            decay * stochastic_noise * random + 1.0
        };

        // So it can shuffle without conflicts
        let mut mutable_training = training_data.clone();

        // Add cost on specific epochs then average and print
        let mut cost = 0.0;
        
        // Loop over all shuffled data `epochs` times
        for epoch in 0..epochs {
            mutable_training.shuffle(&mut shuffle_rng);

            for [input, expected_output] in training_data.iter() {
                self.test(input);

                // Node< Vec< Changes > >
                let mut optimal_activations = expected_output.to_owned();

                // Backpropagate until input layer is reached
                for temp in 1..self.layers.len() {
                    let index = self.layers.len() - temp;
                    let nodes = self.get_layer(index);

                    if self.epochs_per_print != 0 && (epoch + 1) % self.epochs_per_print == 0 && temp == 1 {
                        cost += optimal_activations.iter().enumerate().map(|(i, a)|
                            (a - nodes[i].activation()).powi(2)
                        ).sum::<f64>();
                    }

                    // Changes to activation
                    let inputs: Vec<f64> = self.get_layer(index-1).iter().map(|x| x.activation()).collect();
                    let mut activation_changes: Vec<f64> = inputs.iter().map(|_| 0.0).collect();

                    // Backpropagate each node
                    for i in self.layers[index-1]..self.layers[index] {
                        let changes = self.nodes[i].backpropagate(&inputs, optimal_activations[i-self.layers[index-1]]);
                        
                        // Request changes to previous activations
                        for (input_index, requested) in changes.iter().enumerate() {
                            activation_changes[input_index] +=  *requested * self.learning_rate;
                        }
                    }

                    // Average requested activations and push to vector
                    if index > 1 {
                        optimal_activations = activation_changes.iter().enumerate().map(|(i, c)|
                            inputs[i] - c / self.layers[index] as f64 * noise(epoch)
                        ).collect();
                    }
                }

                for node in self.nodes[self.layers[0]..].iter_mut() {
                    node.apply_changes();
                }
            }

            if self.epochs_per_print != 0 && (epoch + 1) % self.epochs_per_print == 0 {
                println!("Epoch {}/{}:\n\tCost: {}", epoch + 1, epochs, cost / training_data.len() as f64);
                cost = 0.0;
            }
        }
    }
}