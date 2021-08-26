use std::process;

struct Node {
    weights: Vec<f32>,
    bias: f32
}
impl Node {
    fn new(prev_layer_size: u16, bias: f32) -> Self {
        Self {
            weights: (0..prev_layer_size).map(|_| 0.5).collect::<Vec<_>>(),
            bias
        }
    }
}

struct Layer {
    nodes: Vec<Node>,
}
impl Layer {
    fn new(size: u16, prev_layer_size: u16) -> Self {
        let nodes = (0..size).map(|_| Node::new(prev_layer_size, 1.0));
        Self {
            nodes: nodes.collect::<Vec<_>>()
        }
    }

    fn evaluate(&self, input: Vec<f32>) -> Vec<f32> {
        if input.len() != self.nodes[0].weights.len() {
            println!("Given different number of layer inputs than node weights.");
            process::exit(1);
        }

        self.nodes.iter().map(|node| -> f32 {
            let unnormalized = input.iter().enumerate().map(|(index, value)| {
                value * node.weights[index]
            }).sum::<f32>() + node.bias;
            1.0 / (1.0 + 2.0_f32.powf(unnormalized))
        }).collect()
    }
}

struct NeuralNetwork {
    input: u16,
    layers: Vec<Layer>
}
impl NeuralNetwork {
    fn new(input: u16, layers: Vec<u16>) -> Self {
        let layer_list = layers.iter().enumerate().map(|(index, size)| {
            let num_of_weights = match index {
                0 => input,
                _ => layers[index-1]
            };
            Layer::new(*size, num_of_weights)
        });
        Self {
            input,
            layers: layer_list.collect::<Vec<Layer>>()
        }
    }

    fn test(&self, input: Vec<f32>) -> Vec<f32> {
        let mut prev_outputs: Vec<f32> = input;
        for layer in self.layers.iter() {
            prev_outputs = layer.evaluate(prev_outputs);
            println!("{:?}", prev_outputs);
        }
        prev_outputs
    }
}

fn main() {
    let nn = NeuralNetwork::new(4, vec![3, 3, 1]);
    let results = nn.test(vec![2.0, 3.0, 1.0, 8.0]);
    println!("Results: {:?}", results);
}