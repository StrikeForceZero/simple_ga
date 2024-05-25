use rand::Rng;

/// Type alias for all probability / random usages
pub type Odds = f64;

/// Performs a simple coin flip with specified odds of returning true
pub fn coin_flip(odds: Odds) -> bool {
    debug_assert!(
        (0.0..=1.0).contains(&odds),
        "odds must be between 0.0 and 1.0 inclusively, got: {odds}"
    );
    rng::thread_rng().gen_bool(odds)
}

#[derive(Debug, Copy, Clone)]
pub enum Bias {
    Front,
    FrontInverse,
    Back,
    BackInverse,
}

impl Bias {
    pub fn inverse(&self) -> Self {
        match self {
            Self::Front => Self::FrontInverse,
            Self::FrontInverse => Self::Front,
            Self::Back => Self::BackInverse,
            Self::BackInverse => Self::Back,
        }
    }
}

fn bias_value(x: f64, bias: Bias) -> f64 {
    let b = 3f64;
    match bias {
        Bias::Front => {
            let t = x.powf(b);
            let u = 1.0 - (1.0 - x).powf(1.0 / b);
            t + u
        }
        Bias::Back => {
            let t = 1.0 - (x - 1.0).abs().powf(b);
            let u = x.powf(1.0 / b);
            t + u
        }
        Bias::FrontInverse | Bias::BackInverse => 1.0 - bias_value(x, bias.inverse()),
    }
}

/// Returns a random index from 0-len with a given bias
/// x: 0.0 - <1.0
fn _random_index_bias(x: f64, len: usize, bias: Bias) -> usize {
    debug_assert!((0.0..1.0).contains(&x), "x={x} must be between 0.0..1.0");
    let biased_value = bias_value(x, bias) / 2.0;
    // Calculate the index
    (biased_value * len as f64).floor() as usize
}

/// Returns a random index from 0-len with a given bias
pub fn random_index_bias(len: usize, bias: Bias) -> usize {
    let x: f64 = rng::thread_rng().gen_range(0.0..1.0);
    _random_index_bias(x, len, bias)
}

pub mod rng {
    #[cfg(not(test))]
    use rand::prelude::ThreadRng;

    #[cfg(test)]
    pub fn thread_rng() -> simple_ga_internal_lib::test_rng::MockThreadRng {
        simple_ga_internal_lib::test_rng::thread_rng()
    }

    #[cfg(not(test))]
    pub fn thread_rng() -> ThreadRng {
        rand::thread_rng()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest(
        input, expected,
        case::front((100, Bias::Front), |x| x < 30.0),
        case::end((100, Bias::Back), |x| x > 70.0),
    )]
    fn test_random_index_bias(input: (usize, Bias), expected: fn(f32) -> bool) {
        let (input, bias) = input;
        const SAMPLE_SIZE: usize = 10000;
        let avg = (0..SAMPLE_SIZE)
            .map(|n| _random_index_bias(n as f64 / SAMPLE_SIZE as f64, input, bias))
            .sum::<usize>() as f32
            / SAMPLE_SIZE as f32;
        println!("{bias:?} {avg}");
        assert!(expected(avg));
    }
}
