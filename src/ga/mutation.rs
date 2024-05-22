use derivative::Derivative;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;

use crate::ga::fitness::{Fitness, FitnessWrapped};
use crate::ga::population::Population;
use crate::ga::WeightedAction;
use crate::util::{coin_flip, Odds};

#[derive(Derivative, Clone, Default)]
#[derivative(Debug)]
pub struct ApplyMutationOptions<Mutator> {
    pub overall_mutation_chance: Odds,
    /// - `true`: allows each mutation defined to be applied when `P(A∩B)`
    ///     - A: `overall_mutation_chance`
    ///     - B: `Odds` for a given `mutation_actions` entry
    /// - `false`: random mutation is selected from `mutation_actions` based on its Weight (`Odds`)
    pub multi_mutation: bool,
    #[derivative(Debug = "ignore")]
    pub mutation_actions: Vec<WeightedAction<Mutator>>,
    pub clone_on_mutation: bool,
}

pub trait ApplyMutation {
    type Subject;
    fn apply(&self, subject: &Self::Subject) -> Self::Subject;
    fn fitness(subject: &Self::Subject) -> Fitness;
}

pub fn apply_mutations<Mutator: ApplyMutation>(
    population: &mut Population<Mutator::Subject>,
    options: &ApplyMutationOptions<Mutator>,
) {
    let mut rng = thread_rng();
    let mut appended_subjects = vec![];
    for wrapped_subject in population.subjects.iter_mut() {
        if !coin_flip(&mut rng, options.overall_mutation_chance) {
            continue;
        }
        let mut do_mutation = |mutator: &Mutator| {
            let subject = &wrapped_subject.subject();
            let mutated_subject = mutator.apply(subject);
            let fitness = Mutator::fitness(&mutated_subject);
            let fw = FitnessWrapped::new(mutated_subject, fitness);
            if options.clone_on_mutation {
                appended_subjects.push(fw);
            } else {
                *wrapped_subject = fw;
            }
        };
        if options.multi_mutation {
            for weighted_action in options.mutation_actions.iter() {
                if !coin_flip(&mut rng, weighted_action.weight) {
                    continue;
                }
                do_mutation(&weighted_action.action);
            }
        } else {
            let weights: Vec<f64> = options
                .mutation_actions
                .iter()
                .map(|weighted_action| weighted_action.weight)
                .collect();
            if weights.is_empty() {
                continue;
            }
            let dist = WeightedIndex::new(&weights).expect("Weights/Odds should not be all zero");
            let index = dist.sample(&mut rng);
            do_mutation(&options.mutation_actions[index].action);
        }
    }
    population.subjects.extend(appended_subjects);
}
