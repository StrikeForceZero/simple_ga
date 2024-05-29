use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::{
    iter::Rev,
    prelude::*,
    slice::{Iter, IterMut},
};

use crate::ga::fitness::FitnessWrapped;
use crate::ga::prune::PruneRandom;
use crate::ga::select::SelectOtherRandom;
use crate::ga::subject::GaSubject;

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
    pub fn prune_random<P: PruneRandom<Vec<FitnessWrapped<Subject>>>>(&mut self, pruner: P) {
        pruner.prune_random(&mut self.subjects);
    }

    pub fn select_random<'a, S>(&'a self, selector: S) -> S::Output
    where
        S: SelectOtherRandom<&'a FitnessWrapped<Subject>>,
        Subject: 'a,
    {
        selector.select_random(&self.subjects)
    }

    fn _sort(a: &FitnessWrapped<Subject>, b: &FitnessWrapped<Subject>) -> Ordering {
        a.fitness().partial_cmp(&b.fitness()).unwrap()
    }

    fn _sort_rev(a: &FitnessWrapped<Subject>, b: &FitnessWrapped<Subject>) -> Ordering {
        b.fitness().partial_cmp(&a.fitness()).unwrap()
    }

    pub fn add(&mut self, subject: FitnessWrapped<Subject>) {
        self.subjects.push(subject);
    }
}

#[cfg(not(feature = "parallel"))]
impl<Subject> Population<Subject>
where
    Subject: Hash + Eq + PartialEq,
{
    pub fn sort(&mut self) {
        let population = &mut self.subjects;
        population.sort_by(Self::_sort);
    }

    pub fn sort_rev(&mut self) {
        let population = &mut self.subjects;
        population.sort_by(Self::_sort_rev);
    }

    pub fn iter(&self) -> impl Iterator<Item = &FitnessWrapped<Subject>> {
        self.subjects.iter()
    }

    pub fn iter_reverse(&self) -> impl Iterator<Item = &FitnessWrapped<Subject>> {
        self.subjects.iter().rev()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut FitnessWrapped<Subject>> {
        self.subjects.iter_mut()
    }

    pub fn iter_reverse_mut(&mut self) -> impl Iterator<Item = &mut FitnessWrapped<Subject>> {
        self.subjects.iter_mut().rev()
    }
}

#[cfg(feature = "parallel")]
impl<Subject> Population<Subject>
where
    Subject: Hash + Eq + PartialEq,
{
    pub fn sort(&mut self) {
        let population = &mut self.subjects;
        population.par_sort_by(Self::_sort);
    }
    pub fn sort_rev(&mut self) {
        let population = &mut self.subjects;
        population.par_sort_by(Self::_sort_rev);
    }

    pub fn iter(&self) -> Iter<FitnessWrapped<Subject>> {
        self.subjects.par_iter()
    }
    pub fn iter_reverse(&self) -> Rev<Iter<FitnessWrapped<Subject>>> {
        self.iter().rev()
    }

    pub fn iter_mut(&mut self) -> IterMut<FitnessWrapped<Subject>> {
        self.subjects.par_iter_mut()
    }

    pub fn iter_reverse_mut(&mut self) -> Rev<IterMut<FitnessWrapped<Subject>>> {
        self.iter_mut().rev()
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use crate::ga::fitness::{Fitness, FitnessWrapped};
    use crate::ga::population::Population;
    use crate::ga::prune::PruneSingleBackSkipFirst;
    use crate::ga::select::{SelectOther, SelectRandomManyWithBias};
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
            population.prune_random(PruneSingleBackSkipFirst);
            assert_eq!(population.subjects.len(), size - n);
        }
    }

    #[test]
    fn test_generic_select() {
        let population = make_population(2);
        for n in 0..=2 {
            let selected = population.select_random(SelectRandomManyWithBias::new(n, Bias::Front));
            assert_eq!(selected.len(), n);
        }
    }

    #[test]
    fn test_select_front() {
        let population = make_population(2);
        for n in 0..=2 {
            let selected =
                SelectRandomManyWithBias::new(n, Bias::Front).select_from(&population.subjects);
            assert_eq!(selected.len(), n);
        }
    }

    #[test]
    fn test_select_back() {
        let population = make_population(2);
        for n in 0..=2 {
            let selected =
                SelectRandomManyWithBias::new(n, Bias::Back).select_from(&population.subjects);
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
