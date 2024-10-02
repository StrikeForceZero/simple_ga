use std::hash::Hash;

use derivative::Derivative;

use crate::ga::fitness::{Fit, Fitness};
use crate::ga::ga_iterator::{GaIterOptions, GaIterState, GaIterator};
use crate::ga::population::Population;
use crate::ga::subject::GaSubject;
use crate::ga::{GaAction, GaContext, GeneticAlgorithmOptions};

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum GaRunnerCustomForEachGenerationResult {
    Terminate,
}

// TODO: should this be GaIterator? at the expense of requiring the generics to be known at GaRunner construction
type EachGenerationFnOpt<Subject> =
    Option<fn(&mut GaIterState<Subject>) -> Option<GaRunnerCustomForEachGenerationResult>>;

#[derive(Derivative, Clone, Default)]
#[derivative(Debug)]
pub struct GaRunnerOptions<Subject> {
    #[derivative(Debug = "ignore")]
    pub debug_print: Option<fn(&Subject)>,
    pub before_each_generation: EachGenerationFnOpt<Subject>,
    pub after_each_generation: EachGenerationFnOpt<Subject>,
}

pub struct GaRunner<Subject>
where
    Subject: Fit<Fitness> + Hash + PartialEq + Eq,
{
    runner_options: GaRunnerOptions<Subject>,
}

impl<Subject> GaRunner<Subject>
where
    Subject: GaSubject + Fit<Fitness> + Hash + PartialEq + Eq,
{
    pub fn new(runner_options: GaRunnerOptions<Subject>) -> Self {
        Self { runner_options }
    }
    pub fn run<Actions>(
        &mut self,
        ga_options: GeneticAlgorithmOptions<Actions>,
        population: Population<Subject>,
    ) where
        Actions: GaAction<Subject = Subject>,
    {
        #[cfg(test)]
        {
            simple_ga_internal_lib::tracing::init_tracing();
        }
        let mut ga_iter = GaIterator::new_with_options(
            ga_options,
            GaIterState::new(GaContext::default(), population),
            GaIterOptions {
                debug_print: self.runner_options.debug_print,
            },
        );
        while ga_iter.is_fitness_within_range() && !ga_iter.is_fitness_at_target() {
            if let Some(before_each) = self.runner_options.before_each_generation {
                if let Some(result) = before_each(ga_iter.state_mut()) {
                    match result {
                        GaRunnerCustomForEachGenerationResult::Terminate => break,
                    }
                }
            }
            if ga_iter.next_generation().is_none() {
                break;
            }
            if let Some(after_each) = self.runner_options.after_each_generation {
                if let Some(result) = after_each(ga_iter.state_mut()) {
                    match result {
                        GaRunnerCustomForEachGenerationResult::Terminate => break,
                    }
                }
            }
        }
    }
}

pub fn ga_runner<
    Subject: GaSubject + Fit<Fitness> + Hash + PartialEq + Eq,
    Actions: GaAction<Subject = Subject>,
>(
    ga_options: GeneticAlgorithmOptions<Actions>,
    runner_options: GaRunnerOptions<Subject>,
    population: Population<Subject>,
) {
    GaRunner::new(runner_options).run(ga_options, population);
}
