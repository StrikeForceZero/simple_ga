use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;

use itertools::Itertools;
use rand::rngs::ThreadRng;
use tracing::info;
use tracing::log::debug;

use crate::ga::fitness::{Fit, Fitness};
use crate::ga::GeneticAlgorithmOptions;
use crate::ga::mutation::{apply_mutations, ApplyMutation};
use crate::ga::population::Population;
use crate::ga::prune::{PruneExtraSkipFirst, PruneRandom};
use crate::ga::reproduction::{apply_reproductions, ApplyReproduction};

pub struct GaIterState<Subject> {
    pub(crate) generation: usize,
    pub(crate) current_fitness: Fitness,
    pub(crate) reverse_mode_enabled: Option<bool>,
    pub population: Population<Subject>,
}

impl<Subject> GaIterState<Subject> {
    pub fn new(population: Population<Subject>) -> Self {
        Self {
            population,
            generation: 0,
            current_fitness: 0.0,
            reverse_mode_enabled: None,
        }
    }
    pub(crate) fn get_or_determine_reverse_mode_from_options<Mutator, Reproducer, Debug>(&self, options: &GeneticAlgorithmOptions<Mutator, Reproducer, Debug>) -> bool {
        if let Some(reverse_mode_enabled) = self.reverse_mode_enabled {
            reverse_mode_enabled
        } else {
            options.initial_fitness() < options.target_fitness()
        }
    }
    pub(crate) fn get_or_init_reverse_mode_enabled<Mutator, Reproducer, Debug>(&mut self, options: &GeneticAlgorithmOptions<Mutator, Reproducer, Debug>) -> bool {
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

impl<Subject: Debug> Debug for GaIterState<Subject> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GaIterState")
            .field("generation", &self.generation)
            .field("current_fitness", &self.current_fitness)
            .field("reverse_mode_enabled", &self.reverse_mode_enabled)
            .field("population", &self.population)
            .finish()
    }
}

pub struct GaIterator<'rng, Subject, Mutator, Reproducer, Debug> {
    options: GeneticAlgorithmOptions<Mutator, Reproducer, Debug>,
    state: GaIterState<Subject>,
    is_reverse_mode: bool,
    rng: &'rng mut ThreadRng,
}

impl<'rng, Subject, Mutator, Reproducer, Debug> GaIterator<'rng, Subject, Mutator, Reproducer, Debug>
    where
        Subject: Fit<Fitness> + Hash + PartialEq + Eq,
        Mutator: ApplyMutation<Subject=Subject>,
        Reproducer: ApplyReproduction<Subject=Subject>,
        Debug: Fn(&Subject),
{
    pub fn new(
        options: GeneticAlgorithmOptions<Mutator, Reproducer, Debug>,
        mut state: GaIterState<Subject>,
        rng: &'rng mut ThreadRng,
    ) -> Self {
        Self {
            is_reverse_mode: state.get_or_init_reverse_mode_enabled(&options),
            options,
            state,
            rng,
        }
    }

    pub fn state(&self) -> &GaIterState<Subject> {
        &self.state
    }

    pub fn is_fitness_at_target(&self) -> bool {
        self.options.target_fitness() == self.state.current_fitness
    }

    pub fn is_fitness_within_range(&self) -> bool {
        self.options.fitness_range.contains(&self.state.current_fitness)
    }

    pub fn next_generation(&mut self) -> Option<Fitness> {
        self.state.generation += 1;
        let generation_ix = self.state.generation;
        let target_fitness = self.options.target_fitness();
        let current_fitness = self.state.current_fitness;
        if self.state.generation % self.options.log_on_mod_zero_for_generation_ix == 0 {
            info!("generation: {generation_ix}");
        }
        if self.is_reverse_mode {
            self.state.population.sort_rev();
        } else {
            self.state.population.sort();
        }
        if let Some(wrapped_subject) = self.state.population.subjects.first() {
            let subject = wrapped_subject;
            let fitness_will_update = self.is_reverse_mode && subject.fitness() > self.state.current_fitness
                || !self.is_reverse_mode && subject.fitness() < self.state.current_fitness;
            if fitness_will_update {
                self.state.current_fitness = subject.fitness();
                info!("generation: {generation_ix}, fitness: {current_fitness}/{target_fitness}");
                (self.options.debug_print)(&subject.subject())
            }
            if !self.options.fitness_range.contains(&subject.fitness()) {
                debug!(
                    "outside of fitness range: {}..{} ({})",
                    self.options.fitness_range.start,
                    self.options.fitness_range.end,
                    subject.fitness()
                );
                (self.options.debug_print)(&subject.subject());
                return None;
            }
            if self.options.target_fitness() == subject.fitness() {
                debug!("target fitness reached: {target_fitness}");
                (self.options.debug_print)(&subject.subject());
                return None;
            }
        }
        PruneExtraSkipFirst::new(self.state.population.pool_size)
            .prune_random(&mut self.state.population.subjects, self.rng);
        apply_reproductions(&mut self.state.population, &self.options.reproduction_options);
        apply_mutations(&mut self.state.population, &self.options.mutation_options);
        if self.options.remove_duplicates {
            let mut indexes_to_delete: HashSet<usize> = HashSet::new();
            for (a_ix, a_subject) in self.state.population.subjects.iter().enumerate() {
                if indexes_to_delete.contains(&a_ix) {
                    continue;
                }
                for (b_ix, b_subject) in self.state.population.subjects.iter().enumerate() {
                    if a_ix == b_ix || indexes_to_delete.contains(&b_ix) {
                        continue;
                    }
                    if b_subject.fitness() == a_subject.fitness()
                        && b_subject.subject() == a_subject.subject()
                    {
                        indexes_to_delete.insert(b_ix);
                    }
                }
            }
            for ix in indexes_to_delete.iter().sorted().rev() {
                if self.state.population.subjects.len() <= self.state.population.pool_size {
                    break;
                }
                self.state.population.subjects.remove(*ix);
            }
        }
        Some(self.state.current_fitness)
    }
}
