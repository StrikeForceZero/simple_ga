use crate::ga::fitness::{Fit, Fitness};
use crate::ga::population::Population;
use crate::ga::subject::GaSubject;
use crate::ga::{GaAction, GaContext};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Deref;

pub trait InflateTarget {
    type Params;
    type Target;
    type Data;
    fn inflate(&self, params: &Self::Params, target: &mut Self::Target, data: &mut Self::Data);
}

#[derive(Debug, Copy, Clone, Default)]
pub struct InflateUntilFull<Subject, CreateSubjectFunc, Data> {
    create_subject_fn: CreateSubjectFunc,
    _subject_marker: PhantomData<Subject>,
    _data_marker: PhantomData<Data>,
}

impl<Subject, CreateSubjectFunc, Data> InflateUntilFull<Subject, CreateSubjectFunc, Data> {
    pub fn new(create_subject_fn: CreateSubjectFunc) -> Self {
        Self {
            create_subject_fn,
            _subject_marker: PhantomData,
            _data_marker: PhantomData,
        }
    }
}

impl<Subject, CreateSubjectFunc, Data> InflateTarget
    for InflateUntilFull<Subject, CreateSubjectFunc, Data>
where
    Subject: GaSubject + Hash + Eq + PartialEq + Fit<Fitness>,
    CreateSubjectFunc: Fn(&GaContext, &mut Data) -> Subject,
{
    type Params = GaContext;
    type Target = Population<Subject>;
    type Data = Data;
    fn inflate(&self, params: &Self::Params, target: &mut Self::Target, data: &mut Self::Data) {
        while target.subjects.len() < target.pool_size {
            // TODO: re-add warning if fitness = target fitness
            // this would require GeneticAlgorithmOptions to be passed in
            target.add((self.create_subject_fn)(params, data).into());
        }
    }
}

impl<Subject, CreateSubjectFunc, Data> GaAction
    for InflateUntilFull<Subject, CreateSubjectFunc, Data>
where
    Subject: GaSubject + Hash + Eq + PartialEq + Fit<Fitness>,
    CreateSubjectFunc: Fn(&GaContext, &mut Data) -> Subject,
{
    type Subject = Subject;
    type Data = Data;

    fn perform_action(
        &self,
        context: &GaContext,
        population: &mut Population<Self::Subject>,
        data: &mut Self::Data,
    ) {
        self.inflate(context, population, data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
    struct TestSubject;
    impl Fit<Fitness> for TestSubject {
        fn measure(&self) -> Fitness {
            0.0
        }
    }
    impl GaSubject for TestSubject {}

    #[test]
    fn create_subject_fn_pointer() {
        fn create_subject(_: &GaContext) -> TestSubject {
            TestSubject
        }
        let inflate = InflateUntilFull(create_subject);
        let mut population = Population::empty(100);
        inflate.perform_action(&GaContext::default(), &mut population);
        assert_eq!(population.subjects.len(), population.pool_size);
    }
}
