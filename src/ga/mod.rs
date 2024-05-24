use std::cell::{RefCell, RefMut};
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::usize;

use derivative::Derivative;
use rand::Rng;

use population::Population;

use crate::ga::fitness::{Fit, Fitness, FitnessWrapped};
use crate::ga::mutation::ApplyMutationOptions;
use crate::ga::reproduction::ApplyReproductionOptions;
use crate::util::Odds;

pub mod fitness;
pub mod ga_iterator;
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

pub fn create_population_pool<'rng, Subject: Fit<Fitness>, RandNumGen: Rng>(
    rng: &'rng mut RandNumGen,
    options: CreatePopulationOptions<impl Fn(&GaContext<'rng, RandNumGen>) -> Subject>,
) -> Population<Subject> {
    let mut subjects: Vec<FitnessWrapped<Subject>> = vec![];
    let context = GaContext::new(rng);
    for _ in 0..options.population_size {
        let subject = (options.create_subject_fn)(&context);
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
pub struct GeneticAlgorithmOptions<CreateSubjectFn, Mutator, Reproducer> {
    pub remove_duplicates: bool,
    /// initial fitness to target fitness
    pub fitness_initial_to_target_range: Range<Fitness>,
    /// min and max fitness range to terminate the loop
    pub fitness_range: Range<Fitness>,
    pub create_subject_fn: CreateSubjectFn,
    pub cull_amount: usize,
    pub mutation_options: ApplyMutationOptions<Mutator>,
    pub reproduction_options: ApplyReproductionOptions<Reproducer>,
}

impl<CreateSubjectFn, Mutator, Reproducer>
    GeneticAlgorithmOptions<CreateSubjectFn, Mutator, Reproducer>
{
    pub fn initial_fitness(&self) -> Fitness {
        self.fitness_initial_to_target_range.start
    }
    pub fn target_fitness(&self) -> Fitness {
        self.fitness_initial_to_target_range.end
    }
}

pub struct GaContext<'rng, RandNumGen>
where
    RandNumGen: Rng,
{
    pub(self) rng: RefCell<&'rng mut RandNumGen>,
    pub generation: usize,
}

impl<'rng, RandNumGen> GaContext<'rng, RandNumGen>
where
    RandNumGen: Rng,
{
    pub fn new(rng: &'rng mut RandNumGen) -> Self {
        Self {
            rng: RefCell::new(rng),
            generation: 0,
        }
    }
    pub fn rng(&self) -> RefMut<&'rng mut RandNumGen> {
        self.rng.borrow_mut()
    }
    pub fn generation(&self) -> usize {
        self.generation
    }
}
