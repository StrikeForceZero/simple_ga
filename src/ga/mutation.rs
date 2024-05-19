use rand::thread_rng;

use crate::ga::fitness::{Fitness, FitnessWrapped};
use crate::ga::population::Population;
use crate::util::{coin_flip, Odds};

pub struct ApplyMutationOptions<Mutator> {
    pub overall_mutation_chance: Odds,
    pub mutation_chance_tuples: Vec<(Mutator, Odds)>,
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
        for (mutator, odds) in options.mutation_chance_tuples.iter() {
            if !coin_flip(&mut rng, *odds) {
                continue;
            }
            let subject = &wrapped_subject.subject;
            let mutated_subject = mutator.apply(subject);
            let fitness = Mutator::fitness(&mutated_subject);
            let fw = FitnessWrapped::new(mutated_subject, fitness);
            if options.clone_on_mutation {
                appended_subjects.push(fw);
            } else {
                *wrapped_subject = fw;
            }
        }
    }
    population.subjects.extend(appended_subjects);
}
