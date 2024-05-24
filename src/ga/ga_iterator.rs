use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;

use derivative::Derivative;
use itertools::Itertools;
use rand::Rng;
use tracing::info;
use tracing::log::{debug, warn};

use crate::ga::fitness::{Fit, Fitness, FitnessWrapped};
use crate::ga::GeneticAlgorithmOptions;
use crate::ga::mutation::{apply_mutations, ApplyMutation};
use crate::ga::population::Population;
use crate::ga::prune::{PruneExtraSkipFirst, PruneRandom};
use crate::ga::reproduction::{apply_reproductions, ApplyReproduction};

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

pub struct GaIterState<Subject> {
    pub(crate) generation: usize,
    pub(crate) current_fitness: Option<Fitness>,
    pub(crate) reverse_mode_enabled: Option<bool>,
    pub population: Population<Subject>,
}

impl<Subject> GaIterState<Subject> {
    pub fn new(population: Population<Subject>) -> Self {
        Self {
            population,
            generation: 0,
            current_fitness: None,
            reverse_mode_enabled: None,
        }
    }
    pub(crate) fn get_or_determine_reverse_mode_from_options<
        CreateSubjectFn,
        Mutator,
        Reproducer,
    >(
        &self,
        options: &GeneticAlgorithmOptions<CreateSubjectFn, Mutator, Reproducer>,
    ) -> bool {
        if let Some(reverse_mode_enabled) = self.reverse_mode_enabled {
            reverse_mode_enabled
        } else {
            options.initial_fitness() < options.target_fitness()
        }
    }
    pub(crate) fn get_or_init_reverse_mode_enabled<CreateSubjectFn, Mutator, Reproducer>(
        &mut self,
        options: &GeneticAlgorithmOptions<CreateSubjectFn, Mutator, Reproducer>,
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

pub struct GaIterator<'rng, RandNumGen, Subject, CreateSubjectFn, Mutator, Reproducer> {
    options: GeneticAlgorithmOptions<CreateSubjectFn, Mutator, Reproducer>,
    state: GaIterState<Subject>,
    is_reverse_mode: bool,
    ga_iter_options: GaIterOptions<Subject>,
    rng: &'rng mut RandNumGen,
}

impl<'rng, RandNumGen, Subject, CreateSubjectFn, Mutator, Reproducer>
    GaIterator<'rng, RandNumGen, Subject, CreateSubjectFn, Mutator, Reproducer>
where
    RandNumGen: Rng,
    Subject: Fit<Fitness> + Hash + PartialEq + Eq,
    CreateSubjectFn: Fn(&mut RandNumGen, usize) -> Subject,
    Mutator: ApplyMutation<Subject = Subject>,
    Reproducer: ApplyReproduction<Subject = Subject>,
{
    pub fn new(
        options: GeneticAlgorithmOptions<CreateSubjectFn, Mutator, Reproducer>,
        mut state: GaIterState<Subject>,
        rng: &'rng mut RandNumGen,
    ) -> Self {
        Self {
            ga_iter_options: GaIterOptions::default(),
            is_reverse_mode: state.get_or_init_reverse_mode_enabled(&options),
            options,
            state,
            rng,
        }
    }

    pub fn new_with_options(
        options: GeneticAlgorithmOptions<CreateSubjectFn, Mutator, Reproducer>,
        state: GaIterState<Subject>,
        rng: &'rng mut RandNumGen,
        ga_iter_options: GaIterOptions<Subject>,
    ) -> Self {
        let mut iter = Self::new(options, state, rng);
        iter.ga_iter_options = ga_iter_options;
        iter
    }

    pub fn state(&self) -> &GaIterState<Subject> {
        &self.state
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
            debug_print(&subject);
        }
    }

    pub fn next_generation(&mut self) -> Option<Fitness> {
        self.state.generation += 1;
        let generation_ix = self.state.generation;
        let target_fitness = self.options.target_fitness();
        let current_fitness = self.state.current_fitness;
        if self.is_reverse_mode {
            self.state.population.sort_rev();
        } else {
            self.state.population.sort();
        }
        if let Some(wrapped_subject) = self.state.population.subjects.first() {
            let subject = wrapped_subject;
            let fitness_will_update = self.is_reverse_mode
                && subject.fitness() > self.state.current_fitness.unwrap_or(f64::MIN)
                || !self.is_reverse_mode
                    && subject.fitness() < self.state.current_fitness.unwrap_or(f64::MAX);
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
        PruneExtraSkipFirst::new(self.state.population.pool_size - self.options.cull_amount)
            .prune_random(&mut self.state.population.subjects, self.rng);
        apply_reproductions(
            self.rng,
            generation_ix,
            &mut self.state.population,
            &self.options.reproduction_options,
        );
        apply_mutations(
            self.rng,
            generation_ix,
            &mut self.state.population,
            &self.options.mutation_options,
        );
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
        while self.state.population.subjects.len() < self.state.population.pool_size {
            let new_subject = (&self.options.create_subject_fn)(self.rng, generation_ix);
            let fitness_wrapped: FitnessWrapped<Subject> = new_subject.into();
            if fitness_wrapped.fitness() == self.options.target_fitness() {
                warn!("created a subject that measures at the target fitness {} in generation {generation_ix}", self.options.target_fitness());
            }
            self.state.population.subjects.push(fitness_wrapped);
        }
        self.state.current_fitness
    }
}
