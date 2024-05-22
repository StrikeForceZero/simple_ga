use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::ops::Range;
use std::usize;

use derivative::Derivative;
use itertools::Itertools;
use rand::thread_rng;
use tracing::info;
use tracing::log::debug;

use crate::ga::fitness::{Fit, Fitness};
use crate::ga::mutation::{apply_mutations, ApplyMutation, ApplyMutationOptions};
use crate::ga::population::Population;
use crate::ga::prune::{PruneExtraSkipFirst, PruneRandom};
use crate::ga::reproduction::{apply_reproductions, ApplyReproduction, ApplyReproductionOptions};

#[derive(Derivative, Clone, Default)]
#[derivative(Debug)]
pub struct GenerationLoopOptions<Mutator, Reproducer, Debug> {
    pub remove_duplicates: bool,
    /// initial fitness to target fitness
    pub fitness_initial_to_target_range: Range<Fitness>,
    /// min and max fitness range to terminate the loop
    pub fitness_range: Range<Fitness>,
    pub mutation_options: ApplyMutationOptions<Mutator>,
    pub reproduction_options: ApplyReproductionOptions<Reproducer>,
    #[derivative(Debug = "ignore")]
    pub debug_print: Debug,
    pub log_on_mod_zero_for_generation_ix: usize,
}

#[derive(Clone)]
pub struct GenerationLoopState<Subject> {
    pub population: Population<Subject>,
}

impl<Subject: Debug> Debug for GenerationLoopState<Subject> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerationLoopState")
            .field("population", &self.population)
            .finish()
    }
}

pub fn generation_loop<
    Subject: Fit<Fitness> + Hash + PartialEq + Eq,
    Mutator: ApplyMutation<Subject = Subject>,
    Reproducer: ApplyReproduction<Subject = Subject>,
    Debug: Fn(&Subject),
>(
    options: &GenerationLoopOptions<Mutator, Reproducer, Debug>,
    state: &mut GenerationLoopState<Subject>,
) {
    #[cfg(test)]
    {
        simple_ga_internal_lib::tracing::init_tracing();
    }
    let mut rng = thread_rng();
    let mut generation_ix = 0;
    let initial_fitness = options.fitness_initial_to_target_range.start;
    let mut current_fitness = initial_fitness;
    let target_fitness = options.fitness_initial_to_target_range.end;
    let mut is_reverse_mode = false;
    loop {
        generation_ix += 1;
        if generation_ix % options.log_on_mod_zero_for_generation_ix == 0 {
            info!("generation: {generation_ix}");
        }
        state.population.sort();
        if initial_fitness < target_fitness {
            if !is_reverse_mode {
                debug!("enabling reverse mode");
            }
            state.population.subjects.reverse();
            is_reverse_mode = true;
        }
        if let Some(wrapped_subject) = state.population.subjects.first() {
            let subject = wrapped_subject;
            let fitness_will_update = is_reverse_mode && subject.fitness() > current_fitness
                || !is_reverse_mode && subject.fitness() < current_fitness;
            if fitness_will_update {
                current_fitness = subject.fitness();
                info!("generation: {generation_ix}, fitness: {current_fitness}/{target_fitness}");
                (options.debug_print)(&subject.subject())
            }
            if !options.fitness_range.contains(&subject.fitness()) {
                debug!(
                    "outside of fitness range: {}..{} ({})",
                    options.fitness_range.start,
                    options.fitness_range.end,
                    subject.fitness()
                );
                (options.debug_print)(&subject.subject());
                return;
            }
            if target_fitness == subject.fitness() {
                debug!("target fitness reached: {target_fitness}");
                (options.debug_print)(&subject.subject());
                return;
            }
        }
        PruneExtraSkipFirst::new(state.population.pool_size)
            .prune_random(&mut state.population.subjects, &mut rng);
        apply_reproductions(&mut state.population, &options.reproduction_options);
        apply_mutations(&mut state.population, &options.mutation_options);
        if options.remove_duplicates {
            let mut indexes_to_delete: HashSet<usize> = HashSet::new();
            for (a_ix, a_subject) in state.population.subjects.iter().enumerate() {
                if indexes_to_delete.contains(&a_ix) {
                    continue;
                }
                for (b_ix, b_subject) in state.population.subjects.iter().enumerate() {
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
                if state.population.subjects.len() <= state.population.pool_size {
                    break;
                }
                state.population.subjects.remove(*ix);
            }
        }
    }
}
