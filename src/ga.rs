use std::collections::HashSet;
use std::fmt;
use std::fmt::Display;
use std::hash::Hash;

use itertools::Itertools;
use rand::rngs::ThreadRng;

use crate::fitness::{Fit, Fitness, FitnessWrapped};
use crate::util::{Bias, random_index_bias};

pub struct Population<Subject> {
    pub pool_size: usize,
    pub subjects: Vec<FitnessWrapped<Subject>>,
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
    pub fn prune(&mut self, rng: &mut ThreadRng) {
        let mut target_index = 0;
        while target_index == 0 {
            target_index = random_index_bias(rng, self.subjects.len(), Bias::End);
        }
        let population = &mut self.subjects;
        population.drain(target_index..target_index + 1);
    }
    pub fn select(&mut self, rng: &mut ThreadRng, limit: usize) -> Vec<&FitnessWrapped<Subject>> {
        let population = &mut self.subjects;
        // TODO: should this care about uniqueness?
        let mut selected = HashSet::new();
        // since the early iterations will have a lot of blanks we need to set a limit of how many attempts of trying to find unique subjects we find.
        let mut max_iter = limit as f64 * 1.1;
        while selected.len() < limit {
            let target_index = random_index_bias(rng, population.len(), Bias::Front);
            let Some(subject) = population.get(target_index) else {
                panic!(
                    "index miss, tried getting {target_index} with len of {}",
                    population.len()
                );
            };
            selected.insert(subject);
            if max_iter <= 0.0 {
                break;
            }
            max_iter -= 1.0;
        }
        selected.iter().cloned().collect()
    }
    pub fn sort(&mut self) {
        let population = &mut self.subjects;
        population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
    }
    pub fn add(&mut self, subject: FitnessWrapped<Subject>) {
        self.subjects.push(subject);
    }
}

pub type CreateSubjectFn<Subject> = Box<dyn Fn() -> Subject>;

pub struct CreatePopulationOptions<SubjectFn> {
    pub population_size: usize,
    pub create_subject_fn: SubjectFn,
}

pub fn create_population_pool<Subject: Fit<Fitness>>(
    options: CreatePopulationOptions<impl Fn() -> Subject>,
) -> Population<Subject> {
    let mut subjects: Vec<FitnessWrapped<Subject>> = vec![];
    for _ in 0..options.population_size {
        let subject = (options.create_subject_fn)();
        subjects.push(FitnessWrapped::from(subject));
    }
    Population {
        subjects,
        pool_size: options.population_size,
    }
}