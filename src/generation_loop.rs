use std::collections::HashSet;
use std::hash::Hash;
use std::usize;

use itertools::Itertools;
use rand::thread_rng;
use tracing::info;

use crate::fitness::{Fit, Fitness};
use crate::ga::Population;
use crate::mutation::{apply_mutations, ApplyMutation, ApplyMutationOptions};
use crate::reproduction::{apply_reproductions, ApplyReproduction, ApplyReproductionOptions};

pub struct GenerationLoopOptions<Mutator, Reproducer, Debug> {
    pub remove_duplicates: bool,
    pub starting_fitness: Fitness,
    pub target_fitness: Fitness,
    pub min_fitness: Fitness,
    pub max_fitness: Fitness,
    pub mutation_options: ApplyMutationOptions<Mutator>,
    pub reproduction_options: ApplyReproductionOptions<Reproducer>,
    pub debug_print: Debug,
}

pub struct GenerationLoopState<Subject> {
    pub population: Population<Subject>,
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
    let mut rng = thread_rng();
    let mut ix = 0;
    let mut current_fitness = options.starting_fitness;
    loop {
        ix += 1;
        if ix % 100000000 == 0 {
            info!("generation: {}", ix);
        }
        state.population.sort();
        let mut is_reverse_mode = false;
        if options.starting_fitness < options.target_fitness {
            state.population.subjects.reverse();
            is_reverse_mode = true;
        }
        if let Some(wrapped_subject) = state.population.subjects.first() {
            let subject = wrapped_subject;
            let fitness_will_update = is_reverse_mode && subject.fitness > current_fitness
                || !is_reverse_mode && subject.fitness < current_fitness;
            if fitness_will_update {
                current_fitness = subject.fitness;
                info!(
                    "generation: {}, fitness: {}/{}",
                    ix, current_fitness, options.target_fitness
                );
                (options.debug_print)(&subject.subject)
            }
            if subject.fitness >= options.max_fitness
                || subject.fitness <= options.min_fitness
                || options.target_fitness == subject.fitness
            {
                (options.debug_print)(&subject.subject);
                return;
            }
        }
        while state.population.pool_size < state.population.subjects.len() {
            // TODO: should be biased towards the front if we have is_reverse_mode, maybe?
            state.population.prune(&mut rng);
        }
        apply_reproductions(&mut state.population, &options.reproduction_options);
        apply_mutations(&mut state.population, &options.mutation_options);
        if options.remove_duplicates {
            let mut indexes_to_delete: HashSet<usize> = HashSet::new();
            for (ix, wrapped_subject) in state.population.subjects.iter().enumerate() {
                if indexes_to_delete.contains(&ix) {
                    continue;
                }
                for (ix2, wrapped_subject2) in state.population.subjects.iter().enumerate() {
                    if ix == ix2 || indexes_to_delete.contains(&ix2) {
                        continue;
                    }
                    if wrapped_subject2.fitness == wrapped_subject.fitness
                        && wrapped_subject2.subject == wrapped_subject.subject
                    {
                        indexes_to_delete.insert(ix2);
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
