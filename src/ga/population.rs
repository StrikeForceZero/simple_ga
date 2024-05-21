use std::collections::HashSet;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use itertools::Itertools;
use rand::prelude::ThreadRng;

use crate::ga::fitness::FitnessWrapped;
use crate::ga::prune::PruneRandom;
use crate::ga::select::{SelectRandom, SelectRandomManyWithBias};
use crate::util::Bias;

#[derive(Clone)]
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
    pub fn select(&self, rng: &mut ThreadRng, limit: usize) -> Vec<&FitnessWrapped<Subject>> {
        SelectRandomManyWithBias::new(limit, Bias::Front).select_random(rng, &self.subjects)
    }
    pub fn sort(&mut self) {
        let population = &mut self.subjects;
        population.sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());
    }
    pub fn add(&mut self, subject: FitnessWrapped<Subject>) {
        self.subjects.push(subject);
    }
}
