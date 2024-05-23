use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use itertools::Itertools;
use rand::prelude::ThreadRng;

use crate::ga::fitness::FitnessWrapped;
use crate::ga::prune::PruneRandom;
use crate::ga::select::{SelectRandom, SelectRandomManyWithBias};
use crate::util::Bias;

#[derive(Clone, Default)]
pub struct Population<Subject> {
    pub pool_size: usize,
    pub subjects: Vec<FitnessWrapped<Subject>>,
}

impl<Subject: Debug> Debug for Population<Subject> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Population")
            .field("pool_size", &self.pool_size)
            .field("subjects", &self.pool_size)
            .finish()
    }
}

impl<Subject: Display> Display for Population<Subject> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}, {})",
            self.pool_size,
            self.subjects.iter().join(", ")
        )
    }
}

impl<Subject: Hash + Eq + PartialEq> Population<Subject> {
    pub fn prune_random<P: PruneRandom<Vec<FitnessWrapped<Subject>>>>(
        &mut self,
        pruner: P,
        rng: &mut ThreadRng,
    ) {
        pruner.prune_random(&mut self.subjects, rng);
    }
    pub fn select_random<'a, S>(&'a self, rng: &mut ThreadRng, selector: S) -> S::Output
        where
            S: SelectRandom<&'a FitnessWrapped<Subject>>,
            Subject: 'a,
    {
        selector.select_random(rng, &self.subjects)
    }
    pub fn select_front_bias_random(
        &self,
        rng: &mut ThreadRng,
        limit: usize,
    ) -> Vec<&FitnessWrapped<Subject>> {
        SelectRandomManyWithBias::new(limit, Bias::Front).select_random(rng, &self.subjects)
    }
    pub fn select_back_bias_random(
        &self,
        rng: &mut ThreadRng,
        limit: usize,
    ) -> Vec<&FitnessWrapped<Subject>> {
        SelectRandomManyWithBias::new(limit, Bias::Back).select_random(rng, &self.subjects)
    }
    pub fn sort(&mut self) {
        let population = &mut self.subjects;
        population.sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());
    }
    pub fn sort_rev(&mut self) {
        let population = &mut self.subjects;
        population.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());
    }
    pub fn add(&mut self, subject: FitnessWrapped<Subject>) {
        self.subjects.push(subject);
    }

    pub fn iter(&self) -> impl Iterator<Item=&FitnessWrapped<Subject>> {
        self.subjects.iter()
    }

    pub fn iter_reverse(&self) -> impl Iterator<Item=&FitnessWrapped<Subject>> {
        self.subjects.iter().rev()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut FitnessWrapped<Subject>> {
        self.subjects.iter_mut()
    }

    pub fn iter_reverse_mut(&mut self) -> impl Iterator<Item=&mut FitnessWrapped<Subject>> {
        self.subjects.iter_mut().rev()
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rand::thread_rng;

    use crate::ga::fitness::{Fitness, FitnessWrapped};
    use crate::ga::population::Population;
    use crate::ga::prune::PruneSingleSkipFirst;
    use crate::ga::select::SelectRandomManyWithBias;
    use crate::util::Bias;

    fn test_subject(id: u32) -> FitnessWrapped<u32> {
        FitnessWrapped::new(id, id as Fitness)
    }

    impl From<Range<u32>> for Population<u32> {
        fn from(range: Range<u32>) -> Self {
            Population {
                pool_size: range.len(),
                subjects: range.into_iter().map(test_subject).collect(),
            }
        }
    }

    fn make_population(size: usize) -> Population<u32> {
        Population::from(0..size as u32)
    }

    #[test]
    fn test_prune_random() {
        let size = 3;
        let mut population = make_population(size);
        for n in 1..3 {
            population.prune_random(PruneSingleSkipFirst, &mut thread_rng());
            assert_eq!(population.subjects.len(), size - n);
        }
    }

    #[test]
    fn test_generic_select() {
        let population = make_population(2);
        for n in 0..=2 {
            let selected = population.select_random(
                &mut thread_rng(),
                SelectRandomManyWithBias::new(n, Bias::Front),
            );
            assert_eq!(selected.len(), n);
        }
    }

    #[test]
    fn test_select_front() {
        let population = make_population(2);
        for n in 0..=2 {
            let selected = population.select_front_bias_random(&mut thread_rng(), n);
            assert_eq!(selected.len(), n);
        }
    }

    #[test]
    fn test_select_back() {
        let population = make_population(2);
        for n in 0..=2 {
            let selected = population.select_back_bias_random(&mut thread_rng(), n);
            assert_eq!(selected.len(), n);
        }
    }

    #[test]
    fn test_sort() {
        let mut population = make_population(2);
        population.subjects.insert(0, test_subject(3));
        assert_eq!(
            population.subjects,
            vec![test_subject(3), test_subject(0), test_subject(1)]
        );
        population.sort();
        assert_eq!(
            population.subjects,
            vec![test_subject(0), test_subject(1), test_subject(3)]
        );
    }

    #[test]
    fn test_add() {
        let mut population = make_population(0);
        for n in 1..3 {
            population.add(test_subject(n));
            assert_eq!(
                population.subjects,
                (1..=n).map(test_subject).collect::<Vec<_>>()
            );
        }
    }
}
