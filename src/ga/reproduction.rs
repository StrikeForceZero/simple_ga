use std::hash::Hash;

use derivative::Derivative;
use itertools::Itertools;
use rand::thread_rng;

use crate::ga::fitness::{Fitness, FitnessWrapped};
use crate::ga::population::Population;
use crate::util::{coin_flip, Odds};

pub fn asexual_reproduction<Subject: Clone>(subject: &Subject) -> Subject {
    subject.clone()
}

pub type SexualReproductionFn<Subject> = Box<dyn Fn(&Subject, &Subject) -> Subject>;

#[derive(Derivative)]
#[derivative(Debug)]
pub struct ApplyReproductionOptions<Reproducer> {
    pub reproduction_limit: usize,
    pub overall_reproduction_chance: Odds,
    #[derivative(Debug = "ignore")]
    pub reproduction_chance_tuples: Vec<(Reproducer, Odds)>,
}

pub trait ApplyReproduction {
    type Subject: Hash + PartialEq + Eq;
    fn apply(
        &self,
        subject_a: &Self::Subject,
        subject_b: &Self::Subject,
    ) -> (Self::Subject, Self::Subject);
    fn fitness(subject: &Self::Subject) -> Fitness;
}

pub fn apply_reproductions<Reproducer: ApplyReproduction>(
    population: &mut Population<Reproducer::Subject>,
    options: &ApplyReproductionOptions<Reproducer>,
) {
    let mut rng = thread_rng();
    let mut appended_subjects = vec![];
    // TODO: we probably need criteria on who can reproduce with who
    for (subject_a, subject_b) in population
        .select(&mut rng, options.reproduction_limit)
        .iter()
        .tuple_windows()
    {
        if !coin_flip(&mut rng, options.overall_reproduction_chance) {
            continue;
        }
        let (subject_a, subject_b) = (&subject_a.subject(), &subject_b.subject());
        for (reproduction_fn, odds) in options.reproduction_chance_tuples.iter() {
            if !coin_flip(&mut rng, *odds) {
                continue;
            }
            let (offspring_a, offspring_b) = reproduction_fn.apply(subject_a, subject_b);
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
