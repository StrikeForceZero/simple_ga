use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, Range};
use std::rc::Rc;
use std::sync::Arc;
use std::usize;

use crate::ga::fitness::{Fit, Fitness, FitnessWrapped};
use crate::ga::population::Population;
use crate::util::{coin_flip, rng, Odds};
use derivative::Derivative;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;

pub mod action;
pub mod dedupe;
pub mod fitness;
pub mod ga_iterator;
pub mod ga_runner;
pub mod inflate;
pub mod mutation;
pub mod population;
pub mod probability;
pub mod prune;
pub mod reproduction;
pub mod select;
pub mod subject;

#[derive(Derivative, Clone)]
#[derivative(Debug)]
pub struct CreatePopulationOptions<SubjectFn: ?Sized> {
    pub population_size: usize,
    #[derivative(Debug = "ignore")]
    pub create_subject_fn: SubjectFn,
}

pub type CreateSubjectFnArc<Subject, Data> = Arc<dyn Fn(&GaContext<Data>) -> Subject>;
pub type CreateSubjectFnRc<Subject, Data> = Rc<dyn Fn(&GaContext<Data>) -> Subject>;
pub type CreateSubjectFnBox<Subject, Data> = Box<dyn Fn(&GaContext<Data>) -> Subject>;

pub fn create_population_pool<
    Subject: Fit<Fitness>,
    CreateSubjectFn: Deref<Target = dyn Fn(&GaContext<Data>) -> Subject>,
    Data: Default,
>(
    options: CreatePopulationOptions<CreateSubjectFn>,
) -> Population<Subject> {
    let mut subjects: Vec<FitnessWrapped<Subject>> = vec![];
    let mut context = GaContext::default();
    for _ in 0..options.population_size {
        let subject = (options.create_subject_fn)(&mut context);
        subjects.push(FitnessWrapped::from(subject));
    }
    Population {
        subjects,
        pool_size: options.population_size,
    }
}

pub trait SampleSelf {
    type Output;
    fn sample_self(&self) -> Self::Output;
}

// TODO: find new home for docs
/// random action is selected from inner vec based on its Weight (`Odds`)
#[derive(Clone, Default)]
pub struct WeightedActionsSampleOne<Action>(pub Vec<WeightedAction<Action>>);
/// allows each action defined to be applied when `P(A∩B)`
///  - A: `overall_[mutation|reproduction]_chance`
///  - B: `Odds` for a given action entry
#[derive(Clone, Default)]
pub struct WeightedActionsSampleAll<Action>(pub Vec<WeightedAction<Action>>);

// TODO: remove clone?
// TODO: return iterator?
impl<Action: Clone> SampleSelf for WeightedActionsSampleOne<Action> {
    type Output = Vec<Action>;
    fn sample_self(&self) -> Self::Output {
        if self.0.is_empty() {
            return vec![];
        }
        let rng = &mut rng::thread_rng();
        let weights: Vec<f64> = self
            .0
            .iter()
            .map(|weighted_action| weighted_action.weight)
            .collect();
        let dist = WeightedIndex::new(weights).expect("Weights/Odds should not be all zero");
        let index = dist.sample(rng);
        vec![self.0[index].action.clone()]
    }
}

// TODO: remove clone?
// TODO: return iterator?
impl<Action: Clone> SampleSelf for WeightedActionsSampleAll<Action> {
    type Output = Vec<Action>;
    fn sample_self(&self) -> Self::Output {
        if self.0.is_empty() {
            return vec![];
        }
        self.0
            .iter()
            .filter_map(|WeightedAction { action, weight }| {
                if coin_flip(*weight) {
                    Some(action.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Clone, PartialOrd, PartialEq)]
pub struct WeightedAction<Action> {
    pub action: Action,
    pub weight: Odds,
}

impl<Action> Default for WeightedAction<Action>
where
    Action: Default,
{
    fn default() -> Self {
        Self {
            weight: 0.0,
            action: Action::default(),
        }
    }
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
pub struct GeneticAlgorithmOptions<Actions, Data>
where
    Actions: GaAction<Data>,
    Data: Default,
{
    /// initial fitness to target fitness
    pub fitness_initial_to_target_range: Range<Fitness>,
    /// min and max fitness range to terminate the loop
    pub fitness_range: Range<Fitness>,
    pub actions: Actions,
    pub initial_data: Data,
}

impl<Actions, Data> GeneticAlgorithmOptions<Actions, Data>
where
    Actions: GaAction<Data>,
    Data: Default,
{
    pub fn initial_fitness(&self) -> Fitness {
        self.fitness_initial_to_target_range.start
    }
    pub fn target_fitness(&self) -> Fitness {
        self.fitness_initial_to_target_range.end
    }
}

#[derive(Default)]
pub struct GaContext<Data>
where
    Data: Default,
{
    generation: usize,
    pub data: Data,
}

impl<Data> GaContext<Data> {
    pub(crate) fn create_from_data(data: Data) -> Self {
        Self {
            data,
            ..Default::default()
        }
    }
}

impl<Data: Debug> Debug for GaContext<Data>
where
    Data: Default,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GaContext")
            .field("generation", &self.generation)
            .field("data", &self.data)
            .finish()
    }
}

impl<Data> GaContext<Data>
where
    Data: Default,
{
    pub fn generation(&self) -> usize {
        self.generation
    }
    pub(crate) fn increment_generation(&mut self) {
        self.generation += 1;
    }
}

pub trait GaAction<Data: Default> {
    type Subject;
    fn perform_action(&self, context: &GaContext<Data>, population: &mut Population<Self::Subject>);
}
