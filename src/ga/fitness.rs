use std::fmt;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use derivative::Derivative;

pub type Fitness = f64;
pub type CalculateFitnessFn<'a, Subject> = Box<dyn Fn(&Subject) -> Fitness + 'a>;

#[derive(Derivative)]
#[derivative(PartialOrd)]
pub struct FitnessWrapped<Subject> {
    fitness: Fitness,
    #[derivative(PartialEq = "ignore")]
    subject: Rc<Subject>,
}

impl<Subject> FitnessWrapped<Subject> {
    pub fn new(subject: Subject, fitness: Fitness) -> Self {
        FitnessWrapped {
            fitness,
            subject: Rc::new(subject),
        }
    }
    pub fn fitness(&self) -> Fitness {
        self.fitness
    }
    pub fn subject(&self) -> Rc<Subject> {
        self.subject.clone()
    }
}

impl<Subject> From<Subject> for FitnessWrapped<Subject>
where
    Subject: Fit<Fitness>,
{
    fn from(subject: Subject) -> Self {
        let fitness = subject.measure();
        Self::new(subject, fitness)
    }
}

pub trait Fit<Fitness> {
    fn measure(&self) -> Fitness;
}

impl<T> Fit<Fitness> for FitnessWrapped<T> {
    fn measure(&self) -> Fitness {
        self.fitness
    }
}

impl<Subject: Display> Display for FitnessWrapped<Subject> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.fitness, self.subject)
    }
}

impl<Subject: Debug> Debug for FitnessWrapped<Subject> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FitnessWrapped")
            .field("fitness", &self.fitness)
            .field("subject", &self.subject)
            .finish()
    }
}

impl<Subject> Clone for FitnessWrapped<Subject> {
    fn clone(&self) -> Self {
        FitnessWrapped {
            fitness: self.fitness,
            subject: self.subject.clone(),
        }
    }
    fn clone_from(&mut self, source: &Self) {
        self.fitness = source.fitness;
        self.subject = source.subject.clone();
    }
}

impl<Subject: PartialEq> PartialEq<Self> for FitnessWrapped<Subject> {
    fn eq(&self, other: &Self) -> bool {
        self.subject == other.subject
    }
}

impl<Subject: Eq> Eq for FitnessWrapped<Subject> {}

impl<Subject: Hash> Hash for FitnessWrapped<Subject> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.subject.hash(state);
    }
}
