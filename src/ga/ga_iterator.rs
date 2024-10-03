use std::fmt::{Debug, Formatter};
use std::hash::Hash;

use derivative::Derivative;
use tracing::info;
use tracing::log::debug;

use crate::ga::fitness::{Fit, Fitness};
use crate::ga::population::Population;
use crate::ga::subject::GaSubject;
use crate::ga::{EmptyData, GaAction, GaContext, GeneticAlgorithmOptions};

#[derive(Derivative)]
#[derivative(Debug)]
pub struct GaIterOptions<Subject> {
    #[derivative(Debug = "ignore")]
    pub debug_print: Option<fn(&Subject)>,
}

impl<Subject> Default for GaIterOptions<Subject> {
    fn default() -> Self {
        Self { debug_print: None }
    }
}

pub struct GaIterState<Subject, Data = EmptyData> {
    pub(crate) context: GaContext,
    pub(crate) current_fitness: Option<Fitness>,
    pub(crate) reverse_mode_enabled: Option<bool>,
    pub population: Population<Subject>,
    pub data: Data,
}

impl<Subject, Data> GaIterState<Subject, Data> {
    pub fn new(context: GaContext, population: Population<Subject>, data: Data) -> Self {
        Self {
            population,
            context,
            current_fitness: None,
            reverse_mode_enabled: None,
            data,
        }
    }
    pub fn context(&self) -> &GaContext {
        &self.context
    }
    pub(crate) fn get_or_determine_reverse_mode_from_options<
        Actions: GaAction<Subject = Subject, Data = Data>,
    >(
        &self,
        options: &GeneticAlgorithmOptions<Actions, Data>,
    ) -> bool {
        if let Some(reverse_mode_enabled) = self.reverse_mode_enabled {
            reverse_mode_enabled
        } else {
            options.initial_fitness() < options.target_fitness()
        }
    }
    pub(crate) fn get_or_init_reverse_mode_enabled<
        Actions: GaAction<Subject = Subject, Data = Data>,
    >(
        &mut self,
        options: &GeneticAlgorithmOptions<Actions, Data>,
    ) -> bool {
        if let Some(reverse_mode_enabled) = self.reverse_mode_enabled {
            reverse_mode_enabled
        } else {
            let reverse_mode_enabled = self.get_or_determine_reverse_mode_from_options(options);
            if reverse_mode_enabled {
                debug!("enabling reverse mode");
            }
            self.reverse_mode_enabled = Some(reverse_mode_enabled);
            reverse_mode_enabled
        }
    }
}

impl<Subject, Data> Debug for GaIterState<Subject, Data>
where
    Subject: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GaIterState")
            .field("context", &self.context)
            .field("current_fitness", &self.current_fitness)
            .field("reverse_mode_enabled", &self.reverse_mode_enabled)
            .field("population", &self.population)
            .finish()
    }
}

pub struct GaIterator<Subject, Actions, Data = EmptyData>
where
    Actions: GaAction<Subject = Subject, Data = Data>,
{
    options: GeneticAlgorithmOptions<Actions, Data>,
    state: GaIterState<Subject, Data>,
    is_reverse_mode: bool,
    ga_iter_options: GaIterOptions<Subject>,
}

impl<Subject, Actions, Data> GaIterator<Subject, Actions, Data>
where
    Subject: GaSubject + Fit<Fitness> + Hash + PartialEq + Eq,
    Actions: GaAction<Subject = Subject, Data = Data>,
{
    pub fn new(
        options: GeneticAlgorithmOptions<Actions, Data>,
        mut state: GaIterState<Subject, Data>,
    ) -> Self {
        Self {
            ga_iter_options: GaIterOptions::default(),
            is_reverse_mode: state.get_or_init_reverse_mode_enabled(&options),
            options,
            state,
        }
    }

    pub fn new_with_options(
        options: GeneticAlgorithmOptions<Actions, Data>,
        state: GaIterState<Subject, Data>,
        ga_iter_options: GaIterOptions<Subject>,
    ) -> Self {
        let mut iter = Self::new(options, state);
        iter.ga_iter_options = ga_iter_options;
        iter
    }

    pub fn state(&self) -> &GaIterState<Subject, Data> {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut GaIterState<Subject, Data> {
        &mut self.state
    }

    pub fn is_fitness_at_target(&self) -> bool {
        Some(self.options.target_fitness()) == self.state.current_fitness
    }

    pub fn is_fitness_within_range(&self) -> bool {
        let Some(current_fitness) = self.state.current_fitness else {
            return true;
        };
        self.options.fitness_range.contains(&current_fitness)
    }

    pub fn debug_print(&self, subject: &Subject) {
        if let Some(debug_print) = self.ga_iter_options.debug_print {
            debug_print(subject);
        }
    }

    pub fn next_generation(&mut self) -> Option<Fitness> {
        self.state.context.increment_generation();
        let generation_ix = self.state.context.generation;
        let target_fitness = self.options.target_fitness();
        let current_fitness = self.state.current_fitness;
        if self.is_reverse_mode {
            self.state.population.sort_rev();
        } else {
            self.state.population.sort();
        }
        if let Some(wrapped_subject) = self.state.population.subjects.first() {
            let subject = wrapped_subject;
            let fitness_will_update = if self.is_reverse_mode {
                subject.fitness() > self.state.current_fitness.unwrap_or(f64::MIN)
            } else {
                subject.fitness() < self.state.current_fitness.unwrap_or(f64::MAX)
            };
            if fitness_will_update {
                self.state.current_fitness = Some(subject.fitness());
                info!("generation: {generation_ix}, fitness: {current_fitness:?}/{target_fitness}");
                self.debug_print(&subject.subject())
            }
            if !self.options.fitness_range.contains(&subject.fitness()) {
                debug!(
                    "outside of fitness range: {}..{} ({}), generation: {generation_ix}",
                    self.options.fitness_range.start,
                    self.options.fitness_range.end,
                    subject.fitness()
                );
                self.debug_print(&subject.subject());
                return None;
            }
            if self.options.target_fitness() == subject.fitness() {
                debug!("target fitness reached: {target_fitness}, generation: {generation_ix}");
                self.debug_print(&subject.subject());
                return None;
            }
        }

        self.options.actions.perform_action(
            &self.state.context,
            &mut self.state.population,
            &mut self.state.data,
        );
        self.state.current_fitness
    }
}
