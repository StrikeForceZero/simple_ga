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
type EachGenerationFnOpt<Subject, Data> =
    Option<fn(&mut GaIterState<Subject, Data>) -> Option<GaRunnerCustomForEachGenerationResult>>;

#[derive(Derivative, Clone, Default)]
#[derivative(Debug)]
pub struct GaRunnerOptions<Subject, Data>
where
    Data: Default,
{
    #[derivative(Debug = "ignore")]
    pub debug_print: Option<fn(&Subject)>,
    pub before_each_generation: EachGenerationFnOpt<Subject, Data>,
    pub after_each_generation: EachGenerationFnOpt<Subject, Data>,
}

#[derive(Default)]
pub struct GaRunner<Subject, Data>
where
    Subject: Fit<Fitness> + Hash + PartialEq + Eq,
    Data: Default,
{
    runner_options: GaRunnerOptions<Subject, Data>,
}

impl<Subject, Data> GaRunner<Subject, Data>
where
    Subject: GaSubject + Fit<Fitness> + Hash + PartialEq + Eq,
    Data: Default,
{
    pub fn new(runner_options: GaRunnerOptions<Subject, Data>) -> Self {
        Self { runner_options }
    }
    pub fn run<Actions>(
        &mut self,
        ga_options: GeneticAlgorithmOptions<Actions, Data>,
        population: Population<Subject>,
    ) where
        Actions: GaAction<Data, Subject = Subject>,
    {
        #[cfg(test)]
        {
            simple_ga_internal_lib::tracing::init_tracing();
        }
        let mut ga_iter = GaIterator::new_with_options(
            ga_options,
            GaIterState::new(
                GaContext::<Data>::create_from_data(ga_options.initial_data),
                population,
            ),
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
    Actions: GaAction<Data, Subject = Subject>,
    Data: Default,
>(
    ga_options: GeneticAlgorithmOptions<Actions, Data>,
    runner_options: GaRunnerOptions<Subject, Data>,
    population: Population<Subject>,
) {
    GaRunner::new(runner_options).run(ga_options, population);
}
