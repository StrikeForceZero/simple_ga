use std::marker::PhantomData;

use crate::ga::{GaAction, GaContext};
use crate::ga::dedupe::{DedupeAction, DedupeOther};
use crate::ga::fitness::FitnessWrapped;
use crate::ga::mutation::{ApplyMutation, GenericMutator};
use crate::ga::population::Population;
use crate::ga::prune::{PruneAction, PruneOther};
use crate::ga::reproduction::{ApplyReproduction, GenericReproducer};
use crate::ga::select::SelectOther;

#[derive(Debug, Default, Copy, Clone)]
pub struct EmptyAction<Subject>(PhantomData<Subject>);

impl<Subject> GaAction for EmptyAction<Subject> {
    type Subject = ();

    fn perform_action(&self, _context: &GaContext, _population: &mut Population<Self::Subject>) {
        // no op
    }
}

#[derive(Clone)]
pub struct DefaultActions<Subject, Pruner, Mutator, Selector, Reproducer, Dedupe> {
    pub prune: PruneAction<Subject, Pruner>,
    pub mutation: GenericMutator<Mutator, Subject>,
    pub reproduction: GenericReproducer<Reproducer, Selector, Subject>,
    pub dedupe: DedupeAction<Subject, Dedupe>,
}

impl<Subject, Pruner, Mutator, Selector, Reproducer, Dedupe> GaAction
    for DefaultActions<Subject, Pruner, Mutator, Selector, Reproducer, Dedupe>
where
    Pruner: PruneOther<Vec<FitnessWrapped<Subject>>>,
    Mutator: ApplyMutation<Subject = Subject>,
    Selector:
        for<'a> SelectOther<&'a FitnessWrapped<Subject>, Output = Vec<&'a FitnessWrapped<Subject>>>,
    Reproducer: ApplyReproduction<Subject = Subject>,
    Dedupe: DedupeOther<Population<Subject>>,
{
    type Subject = Subject;

    fn perform_action(&self, context: &GaContext, population: &mut Population<Self::Subject>) {
        self.prune.perform_action(context, population);
        self.mutation.perform_action(context, population);
        self.reproduction.perform_action(context, population);
        self.dedupe.perform_action(context, population);
    }
}

impl<Subject, Pruner, Mutator, Selector, Reproducer, Dedupe> Default
    for DefaultActions<Subject, Pruner, Mutator, Selector, Reproducer, Dedupe>
where
    Subject: Default,
    Pruner: Default,
    Mutator: Default,
    Selector: Default,
    Reproducer: Default,
    Dedupe: Default,
{
    fn default() -> Self {
        Self {
            prune: PruneAction::<Subject, Pruner>::default(),
            mutation: GenericMutator::<Mutator, Subject>::default(),
            reproduction: GenericReproducer::<Reproducer, Selector, Subject>::default(),
            dedupe: DedupeAction::<Subject, Dedupe>::default(),
        }
    }
}
