use std::marker::PhantomData;

use derivative::Derivative;

use crate::ga::fitness::{Fitness, FitnessWrapped};
use crate::ga::population::Population;
use crate::ga::subject::GaSubject;
use crate::ga::{GaAction, GaContext, SampleSelf};
use crate::util::{coin_flip, Odds};

#[derive(Clone)]
pub struct GenericMutator<Mutator, Subject, Actions> {
    _subject: PhantomData<Subject>,
    _mutator: PhantomData<Mutator>,
    options: ApplyMutationOptions<Actions>,
}

impl<Mutator, Subject, Actions> GenericMutator<Mutator, Subject, Actions> {
    pub fn new(options: ApplyMutationOptions<Actions>) -> Self {
        Self {
            _subject: PhantomData,
            _mutator: PhantomData,
            options,
        }
    }
}

impl<Mutator, Subject, Actions> Default for GenericMutator<Mutator, Subject, Actions>
where
    Subject: Default,
    Mutator: Default,
    Actions: Default,
{
    fn default() -> Self {
        Self::new(ApplyMutationOptions::<Actions>::default())
    }
}

#[derive(Derivative, Clone, Default)]
#[derivative(Debug)]
pub struct ApplyMutationOptions<Actions> {
    pub overall_mutation_chance: Odds,
    #[derivative(Debug = "ignore")]
    pub mutation_actions: Actions,
    pub clone_on_mutation: bool,
}

pub trait ApplyMutation<Data>
where
    Data: Default,
{
    type Subject: GaSubject;
    fn apply(&self, context: &GaContext<Data>, subject: &Self::Subject) -> Self::Subject;
    fn fitness(subject: &Self::Subject) -> Fitness;
}

pub fn apply_mutations<
    Subject,
    Mutator: ApplyMutation<Data, Subject = Subject>,
    Actions: SampleSelf<Output = Vec<Mutator>>,
    Data: Default,
>(
    context: &GaContext<Data>,
    population: &mut Population<Subject>,
    options: &ApplyMutationOptions<Actions>,
) {
    let mut appended_subjects = vec![];
    for wrapped_subject in population.subjects.iter_mut() {
        if !coin_flip(options.overall_mutation_chance) {
            continue;
        }
        for mutator in options.mutation_actions.sample_self().iter() {
            let subject = &wrapped_subject.subject();
            let mutated_subject = mutator.apply(context, subject);
            let fitness = Mutator::fitness(&mutated_subject);
            let fw = FitnessWrapped::new(mutated_subject, fitness);
            if options.clone_on_mutation {
                appended_subjects.push(fw);
            } else {
                *wrapped_subject = fw;
            }
        }
    }
    population.subjects.extend(appended_subjects);
}

impl<Mutator, Subject, MutatorActions, Data> GaAction<Data>
    for GenericMutator<Mutator, Subject, MutatorActions>
where
    Mutator: ApplyMutation<Data, Subject = Subject>,
    MutatorActions: SampleSelf<Output = Vec<Mutator>>,
    Data: Default,
{
    type Subject = Subject;

    fn perform_action(
        &self,
        context: &GaContext<Data>,
        population: &mut Population<Self::Subject>,
    ) {
        apply_mutations(context, population, &self.options);
    }
}
