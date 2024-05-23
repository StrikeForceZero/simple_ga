use std::hash::Hash;

use derivative::Derivative;
use rand::Rng;
use tracing::info;

use crate::ga::fitness::{Fit, Fitness};
use crate::ga::ga_iterator::{GaIterator, GaIterOptions, GaIterState};
use crate::ga::GeneticAlgorithmOptions;
use crate::ga::mutation::ApplyMutation;
use crate::ga::population::Population;
use crate::ga::reproduction::ApplyReproduction;

#[derive(Derivative, Clone, Default)]
#[derivative(Debug)]
pub struct GaRunnerOptions<Subject> {
    #[derivative(Debug = "ignore")]
    pub debug_print: Option<fn(&Subject)>,
    pub log_on_mod_zero_for_generation_ix: usize,
}

pub struct GaRunner<'rng, RandNumGen, Subject>
where
    RandNumGen: Rng,
    Subject: Fit<Fitness> + Hash + PartialEq + Eq,
{
    runner_options: GaRunnerOptions<Subject>,
    rng: &'rng mut RandNumGen,
}

impl<'rng, RandNumGen, Subject> GaRunner<'rng, RandNumGen, Subject>
where
    RandNumGen: Rng,
    Subject: Fit<Fitness> + Hash + PartialEq + Eq,
{
    pub fn new(runner_options: GaRunnerOptions<Subject>, rng: &'rng mut RandNumGen) -> Self {
        Self {
            runner_options,
            rng,
        }
    }
    pub fn run<Mutator, Reproducer>(
        &mut self,
        ga_options: GeneticAlgorithmOptions<Mutator, Reproducer>,
        population: Population<Subject>,
    ) where
        Mutator: ApplyMutation<Subject = Subject>,
        Reproducer: ApplyReproduction<Subject = Subject>,
    {
        #[cfg(test)]
        {
            simple_ga_internal_lib::tracing::init_tracing();
        }
        let mut ga_iter = GaIterator::new_with_options(
            ga_options,
            GaIterState::new(population),
            self.rng,
            GaIterOptions {
                debug_print: self.runner_options.debug_print,
            },
        );
        while ga_iter.is_fitness_within_range() && !ga_iter.is_fitness_at_target() {
            if ga_iter.state().generation % self.runner_options.log_on_mod_zero_for_generation_ix
                == 0
            {
                info!("generation: {}", ga_iter.state().generation);
            }
            if ga_iter.next_generation().is_none() {
                break;
            }
        }
    }
}

pub fn ga_runner<
    RandNumGen: Rng,
    Subject: Fit<Fitness> + Hash + PartialEq + Eq,
    Mutator: ApplyMutation<Subject = Subject>,
    Reproducer: ApplyReproduction<Subject = Subject>,
>(
    ga_options: GeneticAlgorithmOptions<Mutator, Reproducer>,
    runner_options: GaRunnerOptions<Subject>,
    population: Population<Subject>,
    rng: &mut RandNumGen,
) {
    GaRunner::new(runner_options, rng).run(ga_options, population);
}
