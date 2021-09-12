pub fn constant(rate: f64) -> LearningRate {
    LearningRate::Constant(rate)
}

pub fn momentum(rate: f64, factor: f64) -> LearningRate {
    LearningRate::Momentum(Box::new(move |gradient, previous_gradient| {
        gradient * rate + factor * previous_gradient
    }))
}

pub fn linear_decay(initial_rate: f64) -> LearningRate {
    LearningRate::Decay(Box::new(move |gradient, normalized_epoch| {
        gradient * initial_rate * (1.0 - normalized_epoch)
    }))
}

pub fn inverse_square_root(initial_rate: f64, decay_rate: f64) -> LearningRate {
    LearningRate::SingleParameter(Box::new(move |gradient| gradient.sqrt().recip()))
}

pub enum LearningRate {
    Constant(f64),
    SingleParameter(Box<dyn Fn(f64) -> f64>),
    Momentum(Box<dyn Fn(f64, f64) -> f64>),
    Decay(Box<dyn Fn(f64, f64) -> f64>)
}