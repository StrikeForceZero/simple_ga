use std::hash::Hash;

use crate::ga::{GaAction, GaContext};
use crate::ga::fitness::{Fit, Fitness};
use crate::ga::population::Population;
use crate::ga::subject::GaSubject;

pub trait InflateTarget {
    type Params;
    type Target;
    fn inflate(&self, params: &Self::Params, target: &mut Self::Target);
}

#[derive(Debug, Copy, Clone, Default)]
pub struct InflateUntilFull<F>(pub F);

impl<Subject, CreateSubjectFunc> InflateTarget for InflateUntilFull<CreateSubjectFunc>
where
    Subject: GaSubject + Hash + Eq + PartialEq + Fit<Fitness>,
    CreateSubjectFunc: Fn(&GaContext) -> Subject,
{
    type Params = GaContext;
    type Target = Population<Subject>;
    fn inflate(&self, params: &Self::Params, target: &mut Self::Target) {
        while target.subjects.len() < target.pool_size {
            // TODO: re-add warning if fitness = target fitness
            // this would require GeneticAlgorithmOptions to be passed in
            target.add(self.0(params).into());
        }
    }
}

impl<Subject, CreateSubjectFunc> GaAction for InflateUntilFull<CreateSubjectFunc>
where
    Subject: GaSubject + Hash + Eq + PartialEq + Fit<Fitness>,
    CreateSubjectFunc: Fn(&GaContext) -> Subject,
{
    type Subject = Subject;

    fn perform_action(&self, context: &GaContext, population: &mut Population<Self::Subject>) {
        self.inflate(context, population);
    }
}
