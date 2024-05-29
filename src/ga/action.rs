use std::marker::PhantomData;

use crate::ga::{GaAction, GaContext, SampleSelf};
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
pub struct DefaultActions<
    Subject,
    Pruner,
    MutatorActions,
    Mutator,
    Selector,
    ReproducerActions,
    Reproducer,
    Dedupe,
> {
    pub prune: PruneAction<Subject, Pruner>,
    pub mutation: GenericMutator<Mutator, Subject, MutatorActions>,
    pub reproduction: GenericReproducer<Reproducer, Selector, Subject, ReproducerActions>,
    pub dedupe: DedupeAction<Subject, Dedupe>,
}

impl<Subject, Pruner, MutatorActions, Mutator, Selector, ReproducerActions, Reproducer, Dedupe>
    GaAction
    for DefaultActions<
        Subject,
        Pruner,
        MutatorActions,
        Mutator,
        Selector,
        ReproducerActions,
        Reproducer,
        Dedupe,
    >
where
    Pruner: PruneOther<Vec<FitnessWrapped<Subject>>>,
    Mutator: ApplyMutation<Subject = Subject>,
    MutatorActions: SampleSelf<Output = Vec<Mutator>>,
    Selector:
        for<'a> SelectOther<&'a FitnessWrapped<Subject>, Output = Vec<&'a FitnessWrapped<Subject>>>,
    Reproducer: ApplyReproduction<Subject = Subject>,
    ReproducerActions: SampleSelf<Output = Vec<Reproducer>>,
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

impl<Subject, Pruner, MutatorActions, Mutator, Selector, ReproducerActions, Reproducer, Dedupe>
    Default
    for DefaultActions<
        Subject,
        Pruner,
        MutatorActions,
        Mutator,
        Selector,
        ReproducerActions,
        Reproducer,
        Dedupe,
    >
where
    Subject: Default,
    Pruner: Default,
    MutatorActions: Default,
    Mutator: Default,
    Selector: Default,
    ReproducerActions: Default,
    Reproducer: Default,
    Dedupe: Default,
{
    fn default() -> Self {
        Self {
            prune: PruneAction::<Subject, Pruner>::default(),
            mutation: GenericMutator::<Mutator, Subject, MutatorActions>::default(),
            reproduction:
                GenericReproducer::<Reproducer, Selector, Subject, ReproducerActions>::default(),
            dedupe: DedupeAction::<Subject, Dedupe>::default(),
        }
    }
}
