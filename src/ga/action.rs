use std::marker::PhantomData;

use crate::ga::dedupe::{DedupeAction, DedupeOther};
use crate::ga::fitness::FitnessWrapped;
use crate::ga::inflate::InflateTarget;
use crate::ga::mutation::{ApplyMutation, GenericMutator};
use crate::ga::population::Population;
use crate::ga::prune::{PruneAction, PruneOther};
use crate::ga::reproduction::{ApplyReproduction, GenericReproducer};
use crate::ga::select::SelectOther;
use crate::ga::{GaAction, GaContext, SampleSelf};

#[derive(Debug, Default, Copy, Clone)]
pub struct EmptyAction<Subject>(PhantomData<Subject>);

impl<Subject, Data> GaAction<Data> for EmptyAction<Subject>
where
    Data: Default,
{
    type Subject = ();

    fn perform_action(
        &self,
        _context: &GaContext<Data>,
        _population: &mut Population<Self::Subject>,
    ) {
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
    Inflator,
> {
    pub prune: PruneAction<Subject, Pruner>,
    pub mutation: GenericMutator<Mutator, Subject, MutatorActions>,
    pub reproduction: GenericReproducer<Reproducer, Selector, Subject, ReproducerActions>,
    pub dedupe: DedupeAction<Subject, Dedupe>,
    pub inflate: Inflator,
}

impl<
        Subject,
        Pruner,
        MutatorActions,
        Mutator,
        Selector,
        ReproducerActions,
        Reproducer,
        Dedupe,
        Inflator,
        Data,
    > GaAction<Data>
    for DefaultActions<
        Subject,
        Pruner,
        MutatorActions,
        Mutator,
        Selector,
        ReproducerActions,
        Reproducer,
        Dedupe,
        Inflator,
    >
where
    Pruner: PruneOther<Vec<FitnessWrapped<Subject>>>,
    Mutator: ApplyMutation<Data, Subject = Subject>,
    MutatorActions: SampleSelf<Output = Vec<Mutator>>,
    Selector:
        for<'a> SelectOther<&'a FitnessWrapped<Subject>, Output = Vec<&'a FitnessWrapped<Subject>>>,
    Reproducer: ApplyReproduction<Data, Subject = Subject>,
    ReproducerActions: SampleSelf<Output = Vec<Reproducer>>,
    Dedupe: DedupeOther<Population<Subject>>,
    Inflator: InflateTarget<Params = GaContext<Data>, Target = Population<Subject>>
        + GaAction<Data, Subject = Subject>,
    Data: Default,
{
    type Subject = Subject;

    fn perform_action(
        &self,
        context: &GaContext<Data>,
        population: &mut Population<Self::Subject>,
    ) {
        self.prune.perform_action(context, population);
        self.mutation.perform_action(context, population);
        self.reproduction.perform_action(context, population);
        self.dedupe.perform_action(context, population);
        self.inflate.perform_action(context, population);
    }
}

impl<
        Subject,
        Pruner,
        MutatorActions,
        Mutator,
        Selector,
        ReproducerActions,
        Reproducer,
        Dedupe,
        Inflator,
    > Default
    for DefaultActions<
        Subject,
        Pruner,
        MutatorActions,
        Mutator,
        Selector,
        ReproducerActions,
        Reproducer,
        Dedupe,
        Inflator,
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
    Inflator: Default,
{
    fn default() -> Self {
        Self {
            prune: PruneAction::<Subject, Pruner>::default(),
            mutation: GenericMutator::<Mutator, Subject, MutatorActions>::default(),
            reproduction:
                GenericReproducer::<Reproducer, Selector, Subject, ReproducerActions>::default(),
            dedupe: DedupeAction::<Subject, Dedupe>::default(),
            inflate: Inflator::default(),
        }
    }
}
