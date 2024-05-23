use rand::Rng;

use crate::util::{Bias, random_index_bias};

pub trait PruneRandom<T> {
    fn prune_random<RandNumGen: Rng>(self, items: &mut T, rng: &mut RandNumGen);
}

#[derive(Debug, Copy, Clone)]
/// Used for pruning a single item
pub struct PruneSingleSkipFirst;

impl<T> PruneRandom<Vec<T>> for PruneSingleSkipFirst {
    /// Will randomly remove a single item
    /// Skips the first entry
    fn prune_random<RandNumGen: Rng>(self, items: &mut Vec<T>, rng: &mut RandNumGen) {
        let mut target_index = 0;
        while target_index == 0 {
            target_index = random_index_bias(rng, items.len(), Bias::Back);
        }
        items.drain(target_index..target_index + 1);
    }
}

#[derive(Debug, Copy, Clone)]
/// Used for pruning collections that exceed the max length
pub struct PruneExtraSkipFirst {
    max_length: usize,
}

impl PruneExtraSkipFirst {
    /// Creates a new pruning instance with desired max_length
    pub fn new(max_length: usize) -> Self {
        Self { max_length }
    }
    /// max_length getter
    pub fn max_length(&self) -> usize {
        self.max_length
    }
}

impl<T> PruneRandom<Vec<T>> for PruneExtraSkipFirst {
    /// Will randomly remove items until it reaches the desired length
    /// Skips the first entry
    fn prune_random<RandNumGen: Rng>(self, items: &mut Vec<T>, rng: &mut RandNumGen) {
        while items.len() > self.max_length {
            PruneSingleSkipFirst.prune_random(items, rng);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod single_skip_first {
        use rand::thread_rng;

        use super::*;

        #[test]
        fn test_prune_random() {
            let rng = &mut thread_rng();
            let mut items = vec![1, 2, 3];
            PruneSingleSkipFirst.prune_random(&mut items, rng);
            PruneSingleSkipFirst.prune_random(&mut items, rng);
            assert_eq!(items, vec![1]);
        }
    }

    mod prune_extra_skip_first {
        use rand::thread_rng;

        use super::*;

        #[test]
        fn test_prune_random() {
            let rng = &mut thread_rng();
            let mut items = vec![1, 2, 3];
            PruneExtraSkipFirst::new(1).prune_random(&mut items, rng);
            assert_eq!(items, vec![1]);
        }
    }
}
