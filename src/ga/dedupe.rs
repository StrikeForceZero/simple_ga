use std::hash::Hash;
use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::ga::{GaAction, GaContext};
use crate::ga::population::Population;
use crate::ga::subject::GaSubject;

pub trait DedupeOther<T> {
    fn dedupe(&self, items: &mut T);
}

#[derive(Debug, Copy, Clone)]
pub struct DedupeAction<T, D> {
    _marker: PhantomData<T>,
    action: D,
}

impl<T, D> DedupeAction<T, D> {
    pub fn new(action: D) -> Self {
        Self {
            _marker: PhantomData,
            action,
        }
    }
}

impl<Subject> Default for DedupeAction<Subject, DefaultDedupe<Subject>> {
    fn default() -> Self {
        Self::new(DefaultDedupe::default())
    }
}

impl<Subject, D> GaAction for DedupeAction<Subject, D>
where
    // TODO: this is using population for accessing the para iters when feature is parallel but we might be able to make it target Subject generically
    D: DedupeOther<Population<Subject>>,
{
    type Subject = Subject;

    fn perform_action(&self, _context: &GaContext, population: &mut Population<Self::Subject>) {
        self.action.dedupe(population)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct DefaultDedupe<T> {
    _marker: PhantomData<T>,
}

impl<T> Default for DefaultDedupe<T> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

// TODO: this is using population for accessing the para iters when feature is parallel but we might be able to make it target Vec<Subject>
impl<Subject> DedupeOther<Population<Subject>> for DefaultDedupe<Subject>
where
    Subject: GaSubject + Hash + Eq + PartialEq,
{
    fn dedupe(&self, population: &mut Population<Subject>) {
        #[cfg(feature = "parallel")]
        let indexes_to_delete = {
            use dashmap::DashSet;
            DashSet::new()
        };
        #[cfg(not(feature = "parallel"))]
        let mut indexes_to_delete = {
            use std::collections::HashSet;
            HashSet::new()
        };

        population.iter().enumerate().for_each(|(a_ix, a_subject)| {
            if indexes_to_delete.contains(&a_ix) {
                return;
            }
            population.iter().enumerate().for_each(|(b_ix, b_subject)| {
                if a_ix == b_ix || indexes_to_delete.contains(&b_ix) {
                    return;
                }
                // TODO: should equality check be left to the wrapper struct?
                if b_subject.fitness() == a_subject.fitness()
                    && b_subject.subject() == a_subject.subject()
                {
                    indexes_to_delete.insert(b_ix);
                }
            });
        });
        let indexes_to_delete = {
            #[cfg(feature = "parallel")]
            {
                let mut indexes_to_delete = indexes_to_delete.into_par_iter().collect::<Vec<_>>();
                indexes_to_delete.par_sort_unstable();
                indexes_to_delete
            }
            #[cfg(not(feature = "parallel"))]
            {
                use itertools::Itertools;
                indexes_to_delete.into_iter().sorted()
            }
        };
        for ix in indexes_to_delete.into_iter().rev() {
            // TODO: move this to a generic item remover trait if needed
            // if population.subjects.len() <= population.pool_size {
            //     break;
            // }
            population.subjects.remove(ix);
        }
    }
}
