use std::hash::Hash;
use std::marker::PhantomData;

use derivative::Derivative;
use itertools::Itertools;

use crate::ga::{GaAction, GaContext, SampleSelf};
use crate::ga::fitness::{Fitness, FitnessWrapped};
use crate::ga::mutation::ApplyMutation;
use crate::ga::population::Population;
use crate::ga::select::SelectOther;
use crate::ga::subject::GaSubject;
use crate::util::{coin_flip, Odds};

pub fn asexual_reproduction<Subject: Clone>(subject: &Subject) -> Subject {
    subject.clone()
}

#[derive(Clone)]
pub struct GenericReproducer<Reproducer, Selector, Subject, Actions> {
    _marker: PhantomData<Subject>,
    _reproducer: PhantomData<Reproducer>,
    options: ApplyReproductionOptions<Actions, Selector>,
}

impl<Reproducer, Selector, Subject, Actions>
    GenericReproducer<Reproducer, Selector, Subject, Actions>
{
    pub fn new(options: ApplyReproductionOptions<Actions, Selector>) -> Self {
        Self {
            _marker: PhantomData,
            _reproducer: PhantomData,
            options,
        }
    }
}

impl<Reproducer, Selector, Subject, Actions> Default
    for GenericReproducer<Reproducer, Selector, Subject, Actions>
where
    Subject: Default,
    Reproducer: Default,
    Selector: Default,
    Actions: Default,
{
    fn default() -> Self {
        Self::new(ApplyReproductionOptions::<Actions, Selector>::default())
    }
}

#[derive(Derivative, Clone, Default)]
#[derivative(Debug)]
pub struct ApplyReproductionOptions<Actions, Selector> {
    pub selector: Selector,
    pub overall_reproduction_chance: Odds,
    #[derivative(Debug = "ignore")]
    pub reproduction_actions: Actions,
}

pub trait ApplyReproduction {
    type Subject: GaSubject + Hash + PartialEq + Eq;
    fn apply(
        &self,
        context: &GaContext,
        subject_a: &Self::Subject,
        subject_b: &Self::Subject,
    ) -> (Self::Subject, Self::Subject);
    fn fitness(subject: &Self::Subject) -> Fitness;
}

pub fn apply_reproductions<
    Subject,
    Reproducer: ApplyReproduction<Subject = Subject>,
    Selector: for<'a> SelectOther<&'a FitnessWrapped<Subject>, Output = Vec<&'a FitnessWrapped<Subject>>>,
    Actions: SampleSelf<Output = Vec<Reproducer>>,
>(
    context: &GaContext,
    population: &mut Population<Subject>,
    options: &ApplyReproductionOptions<Actions, Selector>,
) {
    let mut appended_subjects = vec![];
    for (subject_a, subject_b) in options
        .selector
        .select_from(&population.subjects)
        .iter()
        .tuple_windows()
    {
        if !coin_flip(options.overall_reproduction_chance) {
            continue;
        }
        let (subject_a, subject_b) = (&subject_a.subject(), &subject_b.subject());

        for reproducer in options.reproduction_actions.sample_self().iter() {
            let (offspring_a, offspring_b) = reproducer.apply(context, subject_a, subject_b);
            {
                let fitness = Reproducer::fitness(&offspring_a);
                appended_subjects.push(FitnessWrapped::new(offspring_a, fitness));
            }
            {
                let fitness = Reproducer::fitness(&offspring_b);
                appended_subjects.push(FitnessWrapped::new(offspring_b, fitness));
            }
        }
    }
    population.subjects.extend(appended_subjects);
}

impl<Reproducer, Selector, Subject, ReproducerActions> GaAction
    for GenericReproducer<Reproducer, Selector, Subject, ReproducerActions>
where
    Reproducer: ApplyReproduction<Subject = Subject>,
    Selector: for<'a> SelectOther<
        &'a FitnessWrapped<Reproducer::Subject>,
        Output = Vec<&'a FitnessWrapped<Subject>>,
    >,
    ReproducerActions: SampleSelf<Output = Vec<Reproducer>>,
{
    type Subject = Subject;

    fn perform_action(&self, context: &GaContext, population: &mut Population<Self::Subject>) {
        apply_reproductions(context, population, &self.options);
    }
}
