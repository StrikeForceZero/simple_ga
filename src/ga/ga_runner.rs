use std::hash::Hash;

use rand::prelude::ThreadRng;

use crate::ga::fitness::{Fit, Fitness};
use crate::ga::ga_iterator::{GaIterState, GaIterator};
use crate::ga::mutation::ApplyMutation;
use crate::ga::population::Population;
use crate::ga::reproduction::ApplyReproduction;
use crate::ga::GeneticAlgorithmOptions;

pub fn ga_runner<
    Subject: Fit<Fitness> + Hash + PartialEq + Eq,
    Mutator: ApplyMutation<Subject=Subject>,
    Reproducer: ApplyReproduction<Subject=Subject>,
    Debug: Fn(&Subject),
>(
    options: GeneticAlgorithmOptions<Mutator, Reproducer, Debug>,
    population: Population<Subject>,
    rng: &mut ThreadRng,
) {
    #[cfg(test)]
    {
        simple_ga_internal_lib::tracing::init_tracing();
    }
    let state = GaIterState::new(population);
    let mut ga_iter = GaIterator::new(options, state, rng);
    while ga_iter.is_fitness_within_range() && !ga_iter.is_fitness_at_target() {
        if ga_iter.next_generation().is_none() {
            break;
        }
    }
}
