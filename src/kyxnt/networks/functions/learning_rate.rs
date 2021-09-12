pub fn constant(rate: f64) -> LearningRate {
    LearningRate::SingleParameter(Box::new(move |gradient| gradient * rate))
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

pub fn decay(decay_factor: f64, epsilon: f64) -> LearningRate {
    LearningRate::Decay(Box::new(move |gradient, normalized_epoch| {
        gradient / (1.0 + decay_factor * normalized_epoch / epsilon)
    }))
}

pub enum LearningRate {
    SingleParameter(Box<dyn Fn(f64) -> f64>),
    Momentum(Box<dyn Fn(f64, f64) -> f64>),
    Decay(Box<dyn Fn(f64, f64) -> f64>)
}