use derivative::Derivative;
use std::hash::Hash;

use rand::prelude::ThreadRng;
use tracing::info;

use crate::ga::fitness::{Fit, Fitness};
use crate::ga::ga_iterator::{GaIterOptions, GaIterState, GaIterator};
use crate::ga::mutation::ApplyMutation;
use crate::ga::population::Population;
use crate::ga::reproduction::ApplyReproduction;
use crate::ga::GeneticAlgorithmOptions;

#[derive(Derivative, Clone, Default)]
#[derivative(Debug)]
pub struct GaRunnerOptions<Subject> {
    #[derivative(Debug = "ignore")]
    pub debug_print: Option<fn(&Subject)>,
    pub log_on_mod_zero_for_generation_ix: usize,
}

pub fn ga_runner<
    Subject: Fit<Fitness> + Hash + PartialEq + Eq,
    Mutator: ApplyMutation<Subject = Subject>,
    Reproducer: ApplyReproduction<Subject = Subject>,
>(
    ga_options: GeneticAlgorithmOptions<Mutator, Reproducer>,
    runner_options: GaRunnerOptions<Subject>,
    population: Population<Subject>,
    rng: &mut ThreadRng,
) {
    #[cfg(test)]
    {
        simple_ga_internal_lib::tracing::init_tracing();
    }
    let state = GaIterState::new(population);
    let mut ga_iter = GaIterator::new_with_options(
        ga_options,
        state,
        rng,
        GaIterOptions {
            debug_print: runner_options.debug_print,
        },
    );
    while ga_iter.is_fitness_within_range() && !ga_iter.is_fitness_at_target() {
        if ga_iter.state().generation % runner_options.log_on_mod_zero_for_generation_ix == 0 {
            info!("generation: {}", ga_iter.state().generation);
        }
        if ga_iter.next_generation().is_none() {
            break;
        }
    }
}
