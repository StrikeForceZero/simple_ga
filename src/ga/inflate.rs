use crate::ga::fitness::{Fit, Fitness};
use crate::ga::population::Population;
use crate::ga::subject::GaSubject;
use crate::ga::{GaAction, GaContext};
use std::hash::Hash;
use std::ops::Deref;

pub trait InflateTarget {
    type Params;
    type Target;
    fn inflate(&self, params: &Self::Params, target: &mut Self::Target);
}

#[derive(Debug, Copy, Clone, Default)]
pub struct InflateUntilFull<CreateSubjectFunc: ?Sized>(pub CreateSubjectFunc);

impl<Subject, CreateSubjectFunc, Data> InflateTarget for InflateUntilFull<CreateSubjectFunc>
where
    Subject: GaSubject + Hash + Eq + PartialEq + Fit<Fitness>,
    CreateSubjectFunc: Deref<Target = dyn Fn(&GaContext<Data>) -> Subject>,
    Data: Default,
{
    type Params = GaContext<Data>;
    type Target = Population<Subject>;
    fn inflate(&self, params: &Self::Params, target: &mut Self::Target) {
        while target.subjects.len() < target.pool_size {
            // TODO: re-add warning if fitness = target fitness
            // this would require GeneticAlgorithmOptions to be passed in
            target.add((&self.0)(params).into());
        }
    }
}

impl<Subject, CreateSubjectFunc, Data> GaAction<Data> for InflateUntilFull<CreateSubjectFunc>
where
    Subject: GaSubject + Hash + Eq + PartialEq + Fit<Fitness>,
    CreateSubjectFunc: Deref<Target = dyn Fn(&GaContext<Data>) -> Subject>,
    Data: Default,
{
    type Subject = Subject;

    fn perform_action(
        &self,
        context: &GaContext<Data>,
        population: &mut Population<Self::Subject>,
    ) {
        self.inflate(context, population);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
    struct TestSubject;
    impl Fit<Fitness> for TestSubject {
        fn measure(&self) -> Fitness {
            0.0
        }
    }
    impl GaSubject for TestSubject {}

    #[test]
    fn create_subject_arc() {
        let create_subject: Arc<dyn Fn(&GaContext) -> TestSubject> =
            Arc::new(|_: &GaContext| TestSubject);
        let foo = create_subject.clone();
        let create_subject: Box<dyn Fn(&GaContext) -> TestSubject> =
            Box::new(move |ctx: &GaContext| create_subject.clone()(ctx));
        let inflate = InflateUntilFull(create_subject);
        let mut population = Population::empty(100);
        inflate.perform_action(&GaContext::default(), &mut population);
        assert_eq!(population.subjects.len(), population.pool_size);
        assert_eq!(foo(&GaContext::default()), TestSubject);
    }

    #[test]
    fn create_subject_box() {
        let create_subject: Box<dyn Fn(&GaContext) -> TestSubject> =
            Box::new(|_: &GaContext| TestSubject);
        let inflate = InflateUntilFull(create_subject);
        let mut population = Population::empty(100);
        inflate.perform_action(&GaContext::default(), &mut population);
        assert_eq!(population.subjects.len(), population.pool_size);
    }

    #[test]
    fn create_subject_closure() {
        let create_subject: Box<dyn Fn(&GaContext) -> TestSubject> =
            Box::new(|_: &GaContext| -> TestSubject { TestSubject });
        let inflate = InflateUntilFull(create_subject);
        let mut population = Population::empty(100);
        inflate.perform_action(&GaContext::default(), &mut population);
        assert_eq!(population.subjects.len(), population.pool_size);
    }

    #[test]
    fn create_subject_fn_pointer() {
        fn create_subject(_: &GaContext) -> TestSubject {
            TestSubject
        }
        let create_subject: Box<dyn Fn(&GaContext) -> TestSubject> = Box::new(create_subject);
        let inflate = InflateUntilFull(create_subject);
        let mut population = Population::empty(100);
        inflate.perform_action(&GaContext::default(), &mut population);
        assert_eq!(population.subjects.len(), population.pool_size);
    }
}
