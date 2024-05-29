use std::marker::PhantomData;

use crate::ga::{GaAction, GaContext};
use crate::ga::fitness::FitnessWrapped;
use crate::ga::population::Population;
use crate::util::{Bias, random_index_bias};

#[derive(Debug, Default, Copy, Clone)]
pub struct PruneAction<T, P> {
    _marker: PhantomData<T>,
    action: P,
}

impl<T, P> PruneAction<T, P> {
    pub fn new(action: P) -> Self {
        Self {
            _marker: PhantomData,
            action,
        }
    }
}

impl<Subject, P> GaAction for PruneAction<Subject, P>
where
    P: PruneOther<Vec<FitnessWrapped<Subject>>>,
{
    type Subject = Subject;

    fn perform_action(&self, _context: &GaContext, population: &mut Population<Self::Subject>) {
        self.action.prune(&mut population.subjects);
    }
}

pub trait PruneOther<T> {
    fn prune(&self, items: &mut T);
}

pub trait PruneRandom<T> {
    fn prune_random(&self, items: &mut T);
}

#[derive(Debug, Copy, Clone)]
/// Used for pruning a single item
pub struct PruneSingleSkipFirst;

impl<T> PruneOther<Vec<T>> for PruneSingleSkipFirst {
    fn prune(&self, items: &mut Vec<T>) {
        self.prune_random(items);
    }
}

impl<T> PruneRandom<Vec<T>> for PruneSingleSkipFirst {
    /// Will randomly remove a single item
    /// Skips the first entry
    fn prune_random(&self, items: &mut Vec<T>) {
        let mut target_index = 0;
        while target_index == 0 {
            target_index = random_index_bias(items.len(), Bias::Back);
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

impl<T> PruneOther<Vec<T>> for PruneExtraSkipFirst {
    fn prune(&self, items: &mut Vec<T>) {
        self.prune_random(items);
    }
}

impl<T> PruneRandom<Vec<T>> for PruneExtraSkipFirst {
    /// Will randomly remove items until it reaches the desired length
    /// Skips the first entry
    fn prune_random(&self, items: &mut Vec<T>) {
        while items.len() > self.max_length {
            PruneSingleSkipFirst.prune_random(items);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod single_skip_first {
        use super::*;

        #[test]
        fn test_prune_random() {
            let mut items = vec![1, 2, 3];
            PruneSingleSkipFirst.prune_random(&mut items);
            PruneSingleSkipFirst.prune_random(&mut items);
            assert_eq!(items, vec![1]);
        }
    }

    mod prune_extra_skip_first {
        use super::*;

        #[test]
        fn test_prune_random() {
            let mut items = vec![1, 2, 3];
            PruneExtraSkipFirst::new(1).prune_random(&mut items);
            assert_eq!(items, vec![1]);
        }
    }
}
