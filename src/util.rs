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
pub fn random_index_bias(rng: &mut ThreadRng, len: usize, bias: Bias) -> usize {
    let x: f64 = rng.gen_range(0.0..=1.0);

    let biased_value = match bias {
        Bias::Front => 1.0 - x.powf(0.25f64),
        Bias::End => (1.0 - x).powf(0.25f64),
    };

    // Calculate the index
    let value = biased_value * len as f64;
    let value = value.floor() as usize;

    value
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
    use rand::thread_rng;
    use rstest::rstest;

    use super::*;

    #[rstest(
        input, expected,
        case::front((100, Bias::Front), |x| x < 30),
        case::end((100, Bias::End), |x| x > 70),
    )]
    /// There is at worst a 1/10^10 (1 in 10 billion) chance this fails
    fn test_random_index_bias(input: (usize, Bias), expected: fn(usize) -> bool) {
        let (input, bias) = input;
        for _ in 0..100 {
            let avg = (0..input)
                .map(|_| random_index_bias(&mut thread_rng(), input, bias))
                .sum::<usize>()
                / input;
            println!("{bias:?} {avg}");
            assert!(expected(avg));
        }
    }
}
