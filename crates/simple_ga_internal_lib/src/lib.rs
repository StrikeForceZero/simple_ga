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
    use std::cell::UnsafeCell;
    use std::rc::Rc;

    use rand::{CryptoRng, Error, RngCore, SeedableRng};
    use rand::prelude::StdRng;
    use rand::rngs::mock::StepRng;

    pub fn custom_rng(initial: u64, increment: u64) -> StdRng {
        match StdRng::from_rng(StepRng::new(initial, increment)) {
            Ok(rng) => rng,
            Err(err) => panic!("failed to create rng: {err}"),
        }
    }

    pub fn rng() -> StdRng {
        custom_rng(0, 1)
    }

    // below is essentially a near 1:1 copy and paste from rand::rngs::thread
    // https://github.com/rust-random/rand/blob/937320cbfeebd4352a23086d9c6e68f067f74644/src/rngs/thread.rs
    // MIT - https://github.com/rust-random/rand/blob/6a4650691fa1ed8d7c5b0223b408b92ca3a64abd/LICENSE-MIT
    // Copyright 2018 Developers of the Rand project
    // Copyright (c) 2014 The Rust Project Developers

    pub struct MockThreadRng {
        // Rc is explicitly !Send and !Sync
        rng: Rc<UnsafeCell<StdRng>>,
    }

    thread_local!(
        // We require Rc<..> to avoid premature freeing when thread_rng is used
        // within thread-local destructors. See #968.
        static THREAD_RNG_KEY: Rc<UnsafeCell<StdRng>> = {
            Rc::new(UnsafeCell::new(rng()))
        }
    );

    pub fn thread_rng() -> MockThreadRng {
        let rng = THREAD_RNG_KEY.with(|t| t.clone());
        MockThreadRng { rng }
    }

    impl Default for MockThreadRng {
        fn default() -> MockThreadRng {
            thread_rng()
        }
    }

    impl RngCore for MockThreadRng {
        #[inline(always)]
        fn next_u32(&mut self) -> u32 {
            // SAFETY: We must make sure to stop using `rng` before anyone else
            // creates another mutable reference
            let rng = unsafe { &mut *self.rng.get() };
            rng.next_u32()
        }

        #[inline(always)]
        fn next_u64(&mut self) -> u64 {
            // SAFETY: We must make sure to stop using `rng` before anyone else
            // creates another mutable reference
            let rng = unsafe { &mut *self.rng.get() };
            rng.next_u64()
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            // SAFETY: We must make sure to stop using `rng` before anyone else
            // creates another mutable reference
            let rng = unsafe { &mut *self.rng.get() };
            rng.fill_bytes(dest)
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
            // SAFETY: We must make sure to stop using `rng` before anyone else
            // creates another mutable reference
            let rng = unsafe { &mut *self.rng.get() };
            rng.try_fill_bytes(dest)
        }
    }

    impl CryptoRng for MockThreadRng {}

    #[cfg(test)]
    mod test {
        use super::*;

        #[test]
        fn test_thread_rng() {
            use rand::Rng;
            let mut r = thread_rng();
            r.gen::<i32>();
            assert_eq!(r.gen_range(0..1), 0);
        }
    }
}
