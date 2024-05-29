use std::marker::PhantomData;

use crate::ga::{GaAction, GaContext};
use crate::ga::mutation::{ApplyMutation, GenericMutator};
use crate::ga::population::Population;
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
pub struct DefaultActions<Subject, Mutator, Reproducer> {
    pub mutation: GenericMutator<Mutator, Subject>,
    pub reproduction: GenericReproducer<Reproducer, Subject>,
}

impl<Subject, Mutator, Reproducer> GaAction for DefaultActions<Subject, Mutator, Reproducer>
where
    Mutator: ApplyMutation<Subject = Subject>,
    Reproducer: ApplyReproduction<Subject = Subject>,
{
    type Subject = Subject;

    fn perform_action(&self, context: &GaContext, population: &mut Population<Self::Subject>) {
        self.mutation.perform_action(context, population);
        self.reproduction.perform_action(context, population);
    }
}
