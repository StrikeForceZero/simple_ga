use std::hash::{Hash, Hasher};

use derivative::Derivative;

use population::Population;

use crate::ga::fitness::{Fit, Fitness, FitnessWrapped};
use crate::util::Odds;

pub mod fitness;
pub mod generation_loop;
pub mod mutation;
pub mod population;
pub mod probability;
pub mod prune;
pub mod reproduction;
pub mod select;
pub mod subject;

#[derive(Derivative, Clone)]
#[derivative(Debug)]
pub struct CreatePopulationOptions<SubjectFn> {
    pub population_size: usize,
    #[derivative(Debug = "ignore")]
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

#[derive(Clone, PartialOrd, PartialEq)]
pub struct WeightedAction<Action> {
    pub action: Action,
    pub weight: Odds,
}

impl<Action: Hash> Hash for WeightedAction<Action> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.action.hash(state);
        self.weight.to_string().hash(state);
    }
}

impl<Action> From<(Action, Odds)> for WeightedAction<Action> {
    fn from((action, weight): (Action, Odds)) -> Self {
        Self { action, weight }
    }
}
