use population::Population;

use crate::ga::fitness::{Fit, Fitness, FitnessWrapped};

pub mod fitness;
pub mod generation_loop;
pub mod mutation;
pub mod population;
pub mod reproduction;
pub mod subject;

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
