pub mod tracing {
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

pub mod test_rng {
    use rand::prelude::StdRng;
    use rand::rngs::mock::StepRng;
    use rand::SeedableRng;

    pub fn custom_rng(initial: u64, increment: u64) -> StdRng {
        match StdRng::from_rng(StepRng::new(initial, increment)) {
            Ok(rng) => rng,
            Err(err) => panic!("failed to create rng: {err}"),
        }
    }

    pub fn rng() -> StdRng {
        custom_rng(0, 1)
    }
}
