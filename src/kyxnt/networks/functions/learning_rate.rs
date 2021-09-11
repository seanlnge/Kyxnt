pub fn constant(rate: f64) -> LearningRate {
    LearningRate::Constant(rate)
}

pub fn momentum(rate: f64, factor: f64) -> LearningRate {
    LearningRate::Momentum(Box::new(move |gradient, previous_gradient| {
        gradient * rate + factor * previous_gradient
    }))
}

pub enum LearningRate {
    Constant(f64),
    Momentum(Box<dyn Fn(f64, f64) -> f64>)
}