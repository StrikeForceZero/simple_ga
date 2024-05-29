use std::marker::PhantomData;

use crate::ga::{GaAction, GaContext};
use crate::ga::dedupe::{DedupeAction, DedupeOther};
use crate::ga::fitness::FitnessWrapped;
use crate::ga::mutation::{ApplyMutation, GenericMutator};
use crate::ga::population::Population;
use crate::ga::prune::{PruneAction, PruneOther};
use crate::ga::reproduction::{ApplyReproduction, GenericReproducer};

#[derive(Debug, Default, Copy, Clone)]
pub struct EmptyAction<Subject>(PhantomData<Subject>);

impl<Subject> GaAction for EmptyAction<Subject> {
    type Subject = ();

    fn perform_action(&self, _context: &GaContext, _population: &mut Population<Self::Subject>) {
        // no op
    }
}

#[derive(Clone)]
pub struct DefaultActions<Subject, Pruner, Mutator, Reproducer, Dedupe> {
    pub prune: PruneAction<Subject, Pruner>,
    pub mutation: GenericMutator<Mutator, Subject>,
    pub reproduction: GenericReproducer<Reproducer, Subject>,
    pub dedupe: Option<DedupeAction<Subject, Dedupe>>,
}

impl<Subject, Pruner, Mutator, Reproducer, Dedupe> GaAction
    for DefaultActions<Subject, Pruner, Mutator, Reproducer, Dedupe>
where
    Pruner: PruneOther<Vec<FitnessWrapped<Subject>>>,
    Mutator: ApplyMutation<Subject = Subject>,
    Reproducer: ApplyReproduction<Subject = Subject>,
    Dedupe: DedupeOther<Population<Subject>>,
{
    type Subject = Subject;

    fn perform_action(&self, context: &GaContext, population: &mut Population<Self::Subject>) {
        self.prune.perform_action(context, population);
        self.mutation.perform_action(context, population);
        self.reproduction.perform_action(context, population);
        if let Some(dedupe) = &self.dedupe {
            dedupe.perform_action(context, population);
        }
    }
}
