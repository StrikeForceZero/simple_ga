use std::hash::Hash;

use derivative::Derivative;
use itertools::Itertools;
use rand::distributions::{Distribution, WeightedIndex};

use crate::ga::{GaContext, WeightedAction};
use crate::ga::fitness::{Fitness, FitnessWrapped};
use crate::ga::population::Population;
use crate::util::{coin_flip, Odds, rng};

pub fn asexual_reproduction<Subject: Clone>(subject: &Subject) -> Subject {
    subject.clone()
}

#[derive(Derivative, Clone, Default)]
#[derivative(Debug)]
pub struct ApplyReproductionOptions<Reproducer> {
    pub reproduction_limit: usize,
    pub overall_reproduction_chance: Odds,
    /// - `true`: allows each reproduction defined to be applied when `P(A∩B)`
    ///     - A: `overall_reproduction_chance`
    ///     - B: `Odds` for a given `reproduction_actions` entry
    /// - `false`: random reproduction is selected from `reproduction_actions` based on its Weight (`Odds`)
    pub multi_reproduction: bool,
    #[derivative(Debug = "ignore")]
    pub reproduction_actions: Vec<WeightedAction<Reproducer>>,
}

pub trait ApplyReproduction {
    type Subject: Hash + PartialEq + Eq;
    fn apply(
        &self,
        context: &GaContext,
        subject_a: &Self::Subject,
        subject_b: &Self::Subject,
    ) -> (Self::Subject, Self::Subject);
    fn fitness(subject: &Self::Subject) -> Fitness;
}

pub fn apply_reproductions<Reproducer: ApplyReproduction>(
    context: &GaContext,
    population: &mut Population<Reproducer::Subject>,
    options: &ApplyReproductionOptions<Reproducer>,
) {
    let rng = &mut rng::thread_rng();
    let mut appended_subjects = vec![];
    // TODO: we probably need criteria on who can reproduce with who
    for (subject_a, subject_b) in population
        .select_front_bias_random(options.reproduction_limit)
        .iter()
        .tuple_windows()
    {
        if !coin_flip(options.overall_reproduction_chance) {
            continue;
        }
        let (subject_a, subject_b) = (&subject_a.subject(), &subject_b.subject());

        let mut do_reproduction = |reproducer: &Reproducer| {
            let (offspring_a, offspring_b) = reproducer.apply(context, subject_a, subject_b);
            {
                let fitness = Reproducer::fitness(&offspring_a);
                appended_subjects.push(FitnessWrapped::new(offspring_a, fitness));
            }
            {
                let fitness = Reproducer::fitness(&offspring_b);
                appended_subjects.push(FitnessWrapped::new(offspring_b, fitness));
            }
        };

        if options.multi_reproduction {
            for weighted_action in options.reproduction_actions.iter() {
                if !coin_flip(weighted_action.weight) {
                    continue;
                }
                do_reproduction(&weighted_action.action);
            }
        } else {
            let weights: Vec<f64> = options
                .reproduction_actions
                .iter()
                .map(|weighted_action| weighted_action.weight)
                .collect();
            if weights.is_empty() {
                continue;
            }
            let dist = WeightedIndex::new(&weights).expect("Weights/Odds should not be all zero");
            let index = dist.sample(rng);
            do_reproduction(&options.reproduction_actions[index].action);
        }
    }
    population.subjects.extend(appended_subjects);
}
