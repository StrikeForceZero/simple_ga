use rand::Rng;
use rand::rngs::ThreadRng;
use rand_distr::{Distribution, Uniform};

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

pub enum Bias {
    Front,
    End,
}

impl Bias {
    fn to_pos_neg_f64(&self) -> f64 {
        match self {
            Self::Front => 1.0,
            Self::End => -1.0,
        }
    }
}

fn extract_first_decimal(num: f64) -> u8 {
    ((f64::abs(num) * 10.0) % 10.0) as u8
}

fn find_bucket(num: f64, n: usize) -> usize {
    let n = n as f64;
    let num = f64::abs(num);
    (num * n % n) as usize
}

pub fn random_index_bias(rng: &mut ThreadRng, len: usize, bias: Bias) -> usize {
    let lambda = 10.0; // rate parameter
    let uniform = Uniform::new(0.0, 1.0); // uniform distribution on [0,1]
    let y: f64 = uniform.sample(rng);
    // y.ln() is the bias, -y.ln() end / +y.ln() front
    let x: f64 = (bias.to_pos_neg_f64() * y.ln() / lambda) - 1.0;
    let index = find_bucket(x, 10);
    match bias {
        Bias::Front => index,
        Bias::End => len - index - 1,
    }
}

#[cfg(test)]
mod tracing {
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
    use std::collections::HashMap;

    use itertools::Itertools;
    use rand::thread_rng;

    use super::*;

    #[test]
    fn test_cdf() {
        let lambda = 10.0; // rate parameter
        let uniform = Uniform::new(0.0, 1.0); // uniform distribution on [0,1]
        let mut map: HashMap<usize, i32> = HashMap::new();
        let mut rng = thread_rng();
        for _ in 0..10000 {
            let y: f64 = uniform.sample(&mut rng);
            // y.ln() is the bias, -y.ln() end / +y.ln() front
            let x: f64 = (-y.ln() / lambda) - 1.0;
            let index = find_bucket(x, 10);

            let entry = map.entry(index).or_default();
            *entry += 1;
        }
        for (k, v) in map.iter().sorted_by_key(|x| x.0) {
            println!("{k}: {v}");
        }
    }
}
