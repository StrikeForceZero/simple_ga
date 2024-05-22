use std::hash::{Hash, Hasher};

use derivative::Derivative;

use population::Population;
use std::ops::Range;
use std::usize;

use crate::ga::fitness::{Fit, Fitness, FitnessWrapped};
use crate::ga::mutation::ApplyMutationOptions;
use crate::ga::reproduction::ApplyReproductionOptions;
use crate::util::Odds;

pub mod fitness;
pub mod ga_runner;
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

#[derive(Derivative, Clone, Default)]
#[derivative(Debug)]
pub struct GeneticAlgorithmOptions<Mutator, Reproducer, Debug> {
    pub remove_duplicates: bool,
    /// initial fitness to target fitness
    pub fitness_initial_to_target_range: Range<Fitness>,
    /// min and max fitness range to terminate the loop
    pub fitness_range: Range<Fitness>,
    pub mutation_options: ApplyMutationOptions<Mutator>,
    pub reproduction_options: ApplyReproductionOptions<Reproducer>,
    #[derivative(Debug = "ignore")]
    pub debug_print: Debug,
    pub log_on_mod_zero_for_generation_ix: usize,
}
