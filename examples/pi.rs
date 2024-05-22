use std::fmt::{Display, Formatter};

use lazy_static::lazy_static;
use rand::{random, Rng};
use tracing::info;

use simple_ga::ga::{create_population_pool, CreatePopulationOptions};
use simple_ga::ga::fitness::{Fit, Fitness};
use simple_ga::ga::generation_loop::{generation_loop, GenerationLoopOptions, GenerationLoopState};
use simple_ga::ga::mutation::{ApplyMutation, ApplyMutationOptions};
use simple_ga::ga::reproduction::{
    ApplyReproduction, ApplyReproductionOptions, asexual_reproduction,
};

lazy_static! {
    static ref PI_STRING: String = std::f64::consts::PI.to_string();
}

#[derive(Debug, Clone, Default, PartialOrd, PartialEq, Eq, Hash)]
struct Subject(String);

impl Subject {
    fn as_f64(&self) -> f64 {
        self.0
            .parse::<f64>()
            .expect("failed to parse subjects inner f64 from String")
    }
}

impl Display for Subject {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<f64> for Subject {
    fn from(value: f64) -> Self {
        Subject(value.to_string())
    }
}

impl From<Subject> for f64 {
    fn from(value: Subject) -> Self {
        value.as_f64()
    }
}

impl From<&Subject> for f64 {
    fn from(value: &Subject) -> Self {
        value.as_f64()
    }
}

impl Fit<Fitness> for Subject {
    fn measure(&self) -> Fitness {
        let mut fitness = Fitness::default();
        for (ix, c) in self.0.chars().enumerate() {
            if PI_STRING.chars().nth(ix) == Some(c) {
                fitness += 1.0;
            } else {
                break;
            }
        }
        fitness
    }
}

enum MutatorFns {
    AddOne,
    SubOne,
    AddRandom,
    SubRandom,
    Truncate,
    RandTruncate,
}

impl ApplyMutation for MutatorFns {
    type Subject = Subject;

    fn apply(&self, subject: &Self::Subject) -> Self::Subject {
        let subject_f64 = subject.as_f64();
        let mutated_result = match self {
            MutatorFns::AddOne => subject_f64 + 1.0,
            MutatorFns::SubOne => subject_f64 - 1.0,
            MutatorFns::AddRandom => subject_f64 + random_f64(),
            MutatorFns::SubRandom => subject_f64 - random_f64(),
            MutatorFns::Truncate => subject_f64.trunc(),
            MutatorFns::RandTruncate => {
                let offset = rand::thread_rng().gen_range(1..=6) as f64;
                (subject_f64 * offset).trunc() / offset
            }
        };
        Subject(mutated_result.to_string())
    }

    fn fitness(subject: &Self::Subject) -> Fitness {
        subject.measure()
    }
}

enum ReproductionFns {
    SexualGenetic,
    SexualHalf,
    ASexual,
    ZipDecimal,
}

impl ApplyReproduction for ReproductionFns {
    type Subject = Subject;

    fn apply(
        &self,
        subject_a: &Self::Subject,
        subject_b: &Self::Subject,
    ) -> (Self::Subject, Self::Subject) {
        let a = subject_a.as_f64();
        let b = subject_b.as_f64();

        let (a, b) = match self {
            ReproductionFns::SexualGenetic => {
                // TODO: this is wasting hundreds of calculation cycles
                let fitness_a = Self::fitness(subject_a);
                let fitness_b = Self::fitness(subject_b);

                let offspring_a = (a + b) / 2.0;
                let offspring_b = if fitness_a < fitness_b {
                    // prefer a
                    (a + offspring_a) / 2.0
                } else if fitness_b < fitness_a {
                    // prefer b
                    (b + offspring_a) / 2.0
                } else {
                    offspring_a
                };

                (offspring_a, offspring_b)
            }
            ReproductionFns::SexualHalf => {
                let offspring_a = (a + b) / 2.0;
                let offspring_b = (a + b) / 2.0;
                (offspring_a, offspring_b)
            }
            // this is awkward because we are allowing 2 subjects to asexually produce with the same odds
            // and we cant return early, but it saves on iterations
            ReproductionFns::ASexual => (
                asexual_reproduction(subject_a).as_f64(),
                asexual_reproduction(subject_b).as_f64(),
            ),
            ReproductionFns::ZipDecimal => {
                let a_string = a.to_string();
                let b_string = b.to_string();
                let Some((a_left, a_right)) = a_string.split_once('.') else {
                    // no decimal
                    return (a.into(), b.into());
                };
                let Some((b_left, b_right)) = b_string.split_once('.') else {
                    // no decimal
                    return (a.into(), b.into());
                };
                let max_len = a_right.len().max(b_right.len());
                // match right length
                let a_right = format!("{a_right:0>0$}", max_len);
                let b_right = format!("{b_right:0>0$}", max_len);
                let zip = a_right.chars().zip(b_right.chars());
                let (a_right, b_right) = zip.enumerate().fold(
                    (String::new(), String::new()),
                    |(mut a_string, mut b_string), (ix, (a_c, b_c))| {
                        // alternate characters
                        let (a_c, b_c) = if ix % 2 == 0 { (a_c, b_c) } else { (b_c, a_c) };
                        a_string.push(a_c);
                        b_string.push(b_c);
                        (a_string, b_string)
                    },
                );
                // put back together
                let a = format!("{a_left}.{a_right}");
                let b = format!("{b_left}.{b_right}");
                (Subject(a).as_f64(), Subject(b).as_f64())
            }
        };
        (a.into(), b.into())
    }

    fn fitness(subject: &Self::Subject) -> Fitness {
        subject.measure()
    }
}

fn random_f64() -> f64 {
    random::<f64>()
}

fn main() {
    let population_size = 50000;
    simple_ga_internal_lib::tracing::init_tracing();
    let target_fitness = PI_STRING.len() as Fitness;
    fn debug_print(subject: &Subject) {
        let fitness = subject.measure();
        println!("goal: {}", *PI_STRING);
        println!("    :{:01$}^", " ", fitness as usize);
        println!("best: {subject} ({fitness})");
    }
    let generation_loop_options = GenerationLoopOptions {
        remove_duplicates: false,
        fitness_initial_to_target_range: 0f64..target_fitness,
        fitness_range: 0f64..target_fitness,
        mutation_options: ApplyMutationOptions {
            clone_on_mutation: false,
            overall_mutation_chance: 0.10,
            mutation_chance_tuples: vec![
                (MutatorFns::AddOne, 0.25),
                (MutatorFns::SubOne, 0.25),
                (MutatorFns::AddRandom, 0.75),
                (MutatorFns::SubRandom, 0.75),
                (MutatorFns::Truncate, 0.05),
                (MutatorFns::RandTruncate, 0.2),
            ],
        },
        reproduction_options: ApplyReproductionOptions {
            reproduction_limit: population_size / 10,
            overall_reproduction_chance: 1.0,
            reproduction_chance_tuples: vec![
                (ReproductionFns::SexualHalf, 0.50),
                (ReproductionFns::SexualGenetic, 0.75),
                (ReproductionFns::ASexual, 0.10),
                (ReproductionFns::ZipDecimal, 0.60),
            ],
        },
        debug_print,
        log_on_mod_zero_for_generation_ix: 1000000,
    };

    let create_subject_fn = Box::new(|| -> Subject { random_f64().into() });

    let population = create_population_pool(CreatePopulationOptions {
        population_size,
        create_subject_fn: create_subject_fn.clone(),
    });

    let mut generation_loop_state = GenerationLoopState { population };

    info!("starting generation loop");
    generation_loop(&generation_loop_options, &mut generation_loop_state);
    info!("done")
}
