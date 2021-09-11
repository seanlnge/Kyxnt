mod kyxnt;
use kyxnt::networks::*;

use std::fs;
use std::time::Instant;
use serde_json::{Value, from_str};

fn main() {
    let file = fs::read_to_string("./src/training.json").expect("Could not find `training.json` file");
    let unparsed_training_data: Value = from_str(&file).unwrap();

    let parsed_training_data = unparsed_training_data.as_array().unwrap().iter().map(|io| [
        io[0].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f64).collect::<Vec<f64>>(),
        io[1].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f64).collect::<Vec<f64>>(),
    ]).collect::<Vec<[Vec<f64>; 2]>>();

    let start = Instant::now();

    let mut nn = FeedForward::new(2);
    nn.add_layer(3, "sigmoid");
    nn.add_layer(1, "sigmoid");
    
    nn.epochs_per_print = 2000;
    nn.stochastic_noise = 0.0;
    
    //nn.learning_rate = learning_rate::constant(0.5);
    nn.learning_rate = learning_rate::momentum(0.5, 0.75);

    nn.stochastic_gradient_descent(&parsed_training_data, 10000);

    println!("Time: {:?}", start.elapsed());
}