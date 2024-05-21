use itertools::traits::IteratorIndex;
use rand::Rng;
use rand::rngs::ThreadRng;

/// Type alias for all probability / random usages
pub type Odds = f64;

/// Performs a simple coin flip with specified odds of returning true
pub fn coin_flip(rng: &mut ThreadRng, odds: Odds) -> bool {
    debug_assert!(
        (0.0..=1.0).contains(&odds),
        "odds must be between 0.0 and 1.0 inclusively, got: {odds}"
    );
    rng.gen_bool(odds)
}

#[derive(Debug, Copy, Clone)]
pub enum Bias {
    Front,
    End,
}

impl Bias {
    pub fn inverse(&self) -> Self {
        match self {
            Self::Front => Self::End,
            Self::End => Self::Front,
        }
    }
}

fn extract_first_decimal(num: f64) -> u8 {
    ((f64::abs(num) * 10.0) % 10.0) as u8
}

/// Returns a random index from 0-len with a given bias
/// x: 0.0 - <1.0
fn _random_index_bias(x: f64, len: usize, bias: Bias) -> usize {
    debug_assert!((0.0..1.0).contains(&x), "x={x} must be between 0.0..1.0");
    let b = 3f64;
    let biased_value = match bias {
        Bias::Front => {
            let t = x.powf(b);
            let u = 1.0 - (1.0 - x).powf(1.0 / b);
            t + u
        }
        Bias::End => {
            let t = 1.0 - (x - 1.0).abs().powf(b);
            let u = x.powf(1.0 / b);
            t + u
        }
    } / 2.0;

    // Calculate the index
    (biased_value * len as f64).floor() as usize
}

/// Returns a random index from 0-len with a given bias
pub fn random_index_bias(rng: &mut ThreadRng, len: usize, bias: Bias) -> usize {
    let x: f64 = rng.gen_range(0.0..1.0);
    _random_index_bias(x, len, bias)
}

#[cfg(test)]
pub(crate) mod debug_tracing {
    use std::sync::Once;

    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::fmt::format::FmtSpan;

    static TRACING_SUBSCRIBER: Once = Once::new();

    pub fn init_tracing() {
        TRACING_SUBSCRIBER.call_once(|| {
            tracing_subscriber::fmt()
                .with_env_filter(EnvFilter::from_default_env())
                .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
                .with_target(true)
                .init();
            tracing::info!("Info enabled");
            tracing::debug!("Debug enabled");
            tracing::trace!("Trace enabled");
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest(
        input, expected,
        case::front((100, Bias::Front), |x| x < 30.0),
        case::end((100, Bias::End), |x| x > 70.0),
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
