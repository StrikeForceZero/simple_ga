use std::marker::PhantomData;

use crate::ga::fitness::FitnessWrapped;
use crate::ga::population::Population;
use crate::ga::{GaAction, GaContext};
use crate::util::{random_index_bias, Bias};

#[derive(Debug, Default, Copy, Clone)]
pub struct EmptyPrune;

impl<T> PruneOther<T> for EmptyPrune {
    fn prune(&self, _items: &mut T) {
        // no op
    }
}

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
    type Data = ();

    fn perform_action(
        &self,
        _context: &GaContext,
        population: &mut Population<Self::Subject>,
        _data: &mut Self::Data,
    ) {
        self.action.prune(&mut population.subjects);
    }
}

pub trait PruneOther<T> {
    fn prune(&self, items: &mut T);
}

pub trait PruneRandom<T> {
    fn prune_random(&self, items: &mut T);
}

macro_rules! create_sized_prune_skip_first {
    ($name:ident, $amount:literal, $bias:expr) => {
        #[derive(Debug, Copy, Clone, Default)]
        pub struct $name;

        impl<T> PruneOther<Vec<T>> for $name {
            fn prune(&self, items: &mut Vec<T>) {
                self.prune_random(items);
            }
        }

        impl<T> PruneRandom<Vec<T>> for $name {
            /// Will randomly remove a $amount items
            fn prune_random(&self, items: &mut Vec<T>) {
                let target_size = (items.len() as f64 * $amount).round() as usize;
                while items.len() > target_size {
                    match $bias {
                        Bias::Front | Bias::BackInverse => {
                            PruneSingleFrontSkipFirst.prune_random(items);
                        }
                        Bias::Back | Bias::FrontInverse => {
                            PruneSingleBackSkipFirst.prune_random(items);
                        }
                    };
                }
            }
        }
    };
}

macro_rules! create_sized_prune {
    ($name:ident, $amount:literal, $bias:expr) => {
        #[derive(Debug, Copy, Clone)]
        pub struct $name;

        impl<T> PruneOther<Vec<T>> for $name {
            fn prune(&self, items: &mut Vec<T>) {
                self.prune_random(items);
            }
        }

        impl<T> PruneRandom<Vec<T>> for $name {
            /// Will randomly remove a $amount items
            fn prune_random(&self, items: &mut Vec<T>) {
                let target_size = (items.len() as f64 * $amount).round() as usize;
                while items.len() > target_size {
                    match $bias {
                        Bias::Front | Bias::BackInverse => {
                            PruneSingleFront.prune_random(items);
                        }
                        Bias::Back | Bias::FrontInverse => {
                            PruneSingleBack.prune_random(items);
                        }
                    };
                }
            }
        }
    };
}

create_sized_prune!(DefaultPruneHalfBack, 0.5, Bias::Back);
create_sized_prune!(DefaultPruneQuarterBack, 0.25, Bias::Back);
create_sized_prune!(DefaultPruneThreeQuarterBack, 0.75, Bias::Back);
create_sized_prune!(DefaultPruneThirdBack, 0.33, Bias::Back);
create_sized_prune!(DefaultPruneTwoThirdBack, 0.66, Bias::Back);

create_sized_prune!(DefaultPruneHalfFront, 0.5, Bias::Front);
create_sized_prune!(DefaultPruneQuarterFront, 0.25, Bias::Front);
create_sized_prune!(DefaultPruneThreeQuarterFront, 0.75, Bias::Front);
create_sized_prune!(DefaultPruneThirdFront, 0.33, Bias::Front);
create_sized_prune!(DefaultPruneTwoThirdFront, 0.66, Bias::Front);

create_sized_prune_skip_first!(DefaultPruneHalfBackSkipFirst, 0.5, Bias::Back);
create_sized_prune_skip_first!(DefaultPruneQuarterBackSkipFirst, 0.25, Bias::Back);
create_sized_prune_skip_first!(DefaultPruneThreeQuarterBackSkipFirst, 0.75, Bias::Back);
create_sized_prune_skip_first!(DefaultPruneThirdBackSkipFirst, 0.33, Bias::Back);
create_sized_prune_skip_first!(DefaultPruneTwoThirdBackSkipFirst, 0.66, Bias::Back);

create_sized_prune_skip_first!(DefaultPruneHalfFrontSkipFirst, 0.5, Bias::Front);
create_sized_prune_skip_first!(DefaultPruneQuarterFrontSkipFirst, 0.25, Bias::Front);
create_sized_prune_skip_first!(DefaultPruneThreeQuarterFrontSkipFirst, 0.75, Bias::Front);
create_sized_prune_skip_first!(DefaultPruneThirdFrontSkipFirst, 0.33, Bias::Front);
create_sized_prune_skip_first!(DefaultPruneTwoThirdFrontSkipFirst, 0.66, Bias::Front);

#[derive(Debug, Copy, Clone, Default)]
/// Used for pruning a single item from the back
pub struct PruneSingleBack;

impl<T> PruneOther<Vec<T>> for PruneSingleBack {
    fn prune(&self, items: &mut Vec<T>) {
        self.prune_random(items);
    }
}

impl<T> PruneRandom<Vec<T>> for PruneSingleBack {
    /// Will randomly remove a single item from the back
    fn prune_random(&self, items: &mut Vec<T>) {
        let target_index = random_index_bias(items.len(), Bias::Back);
        items.drain(target_index..target_index + 1);
    }
}

#[derive(Debug, Copy, Clone, Default)]
/// Used for pruning a single item from the front
pub struct PruneSingleFront;

impl<T> PruneOther<Vec<T>> for PruneSingleFront {
    fn prune(&self, items: &mut Vec<T>) {
        self.prune_random(items);
    }
}

impl<T> PruneRandom<Vec<T>> for PruneSingleFront {
    /// Will randomly remove a single item from the front
    fn prune_random(&self, items: &mut Vec<T>) {
        let target_index = random_index_bias(items.len(), Bias::Front);
        items.drain(target_index..target_index + 1);
    }
}

#[derive(Debug, Copy, Clone, Default)]
/// Used for pruning a single item from the back, skipping the first
pub struct PruneSingleBackSkipFirst;

impl<T> PruneOther<Vec<T>> for PruneSingleBackSkipFirst {
    fn prune(&self, items: &mut Vec<T>) {
        self.prune_random(items);
    }
}

impl<T> PruneRandom<Vec<T>> for PruneSingleBackSkipFirst {
    /// Will randomly remove a single item from the back
    /// Skips the first entry
    fn prune_random(&self, items: &mut Vec<T>) {
        let mut target_index = 0;
        while target_index == 0 {
            target_index = random_index_bias(items.len(), Bias::Back);
        }
        items.drain(target_index..target_index + 1);
    }
}

#[derive(Debug, Copy, Clone, Default)]
/// Used for pruning a single item from the front, skipping the first
pub struct PruneSingleFrontSkipFirst;

impl<T> PruneOther<Vec<T>> for PruneSingleFrontSkipFirst {
    fn prune(&self, items: &mut Vec<T>) {
        self.prune_random(items);
    }
}

impl<T> PruneRandom<Vec<T>> for PruneSingleFrontSkipFirst {
    /// Will randomly remove a single item from the front
    /// Skips the first entry
    fn prune_random(&self, items: &mut Vec<T>) {
        let mut target_index = 0;
        while target_index == 0 {
            target_index = random_index_bias(items.len(), Bias::Front);
        }
        items.drain(target_index..target_index + 1);
    }
}

#[derive(Debug, Copy, Clone, Default)]
/// Used for pruning collections that exceed the max length
pub struct PruneExtraBackSkipFirst {
    max_length: usize,
}

impl PruneExtraBackSkipFirst {
    /// Creates a new pruning instance with desired max_length
    pub fn new(max_length: usize) -> Self {
        Self { max_length }
    }
    /// max_length getter
    pub fn max_length(&self) -> usize {
        self.max_length
    }
}

impl<T> PruneOther<Vec<T>> for PruneExtraBackSkipFirst {
    fn prune(&self, items: &mut Vec<T>) {
        self.prune_random(items);
    }
}

impl<T> PruneRandom<Vec<T>> for PruneExtraBackSkipFirst {
    /// Will randomly remove items until it reaches the desired length
    /// Skips the first entry
    fn prune_random(&self, items: &mut Vec<T>) {
        while items.len() > self.max_length {
            PruneSingleBackSkipFirst.prune_random(items);
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// Used for pruning collections that exceed the max length
pub struct PruneExtraFrontSkipFirst {
    max_length: usize,
}

impl PruneExtraFrontSkipFirst {
    /// Creates a new pruning instance with desired max_length
    pub fn new(max_length: usize) -> Self {
        Self { max_length }
    }
    /// max_length getter
    pub fn max_length(&self) -> usize {
        self.max_length
    }
}

impl<T> PruneOther<Vec<T>> for PruneExtraFrontSkipFirst {
    fn prune(&self, items: &mut Vec<T>) {
        self.prune_random(items);
    }
}

impl<T> PruneRandom<Vec<T>> for PruneExtraFrontSkipFirst {
    /// Will randomly remove items until it reaches the desired length
    /// Skips the first entry
    fn prune_random(&self, items: &mut Vec<T>) {
        while items.len() > self.max_length {
            PruneSingleFrontSkipFirst.prune_random(items);
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
            PruneSingleBackSkipFirst.prune_random(&mut items);
            PruneSingleBackSkipFirst.prune_random(&mut items);
            assert_eq!(items, vec![1]);
        }
    }

    mod prune_extra_skip_first {
        use super::*;

        #[test]
        fn test_prune_random() {
            let mut items = vec![1, 2, 3];
            PruneExtraBackSkipFirst::new(1).prune_random(&mut items);
            assert_eq!(items, vec![1]);
        }
    }
}
