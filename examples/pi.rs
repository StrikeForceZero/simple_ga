use std::fmt::{Display, Formatter};

use lazy_static::lazy_static;
use rand::Rng;
use tracing::{debug, info};

use simple_ga::ga::{
    create_population_pool, CreatePopulationOptions, GaContext, GeneticAlgorithmOptions,
    WeightedActionsSampleOne,
};
use simple_ga::ga::action::DefaultActions;
use simple_ga::ga::dedupe::EmptyDedupe;
use simple_ga::ga::fitness::{Fit, Fitness};
use simple_ga::ga::ga_runner::{ga_runner, GaRunnerOptions};
use simple_ga::ga::mutation::{ApplyMutation, ApplyMutationOptions, GenericMutator};
use simple_ga::ga::prune::{PruneAction, PruneExtraBackSkipFirst};
use simple_ga::ga::reproduction::{
    ApplyReproduction, ApplyReproductionOptions, asexual_reproduction, GenericReproducer,
};
use simple_ga::ga::select::SelectRandomManyWithBias;
use simple_ga::ga::subject::GaSubject;
use simple_ga::util::{Bias, rng};

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

fn random_pos_add(input: f64, pos: usize, amount: i8) -> f64 {
    let sign = if amount.is_negative() { "-" } else { "" };
    let amount = amount.abs();
    format!("{sign}0.{amount:0>pos$}")
        .parse::<f64>()
        .expect("failed to generate f64")
        + input
}

#[derive(Debug, Copy, Clone, Default)]
enum MutatorFns {
    #[default]
    NoOp,
    AddOne,
    SubOne,
    AddRandom,
    SubRandom,
    AddRandomPosOne,
    SubRandomPosOne,
    Truncate,
    RandTruncate,
}

impl GaSubject for Subject {}

impl ApplyMutation for MutatorFns {
    type Subject = Subject;

    fn apply(&self, _context: &GaContext, subject: &Self::Subject) -> Self::Subject {
        let subject_f64 = subject.as_f64();
        let rng = &mut rng::thread_rng();
        let mutated_result = match self {
            Self::NoOp => panic!("noop mutator fn only used to satisfy default impl"),
            MutatorFns::AddOne => subject_f64 + 1.0,
            MutatorFns::SubOne => subject_f64 - 1.0,
            MutatorFns::AddRandom => subject_f64 + random_f64(rng),
            MutatorFns::SubRandom => subject_f64 - random_f64(rng),
            MutatorFns::AddRandomPosOne => random_pos_add(subject_f64, rng.gen_range(0..16), 1),
            MutatorFns::SubRandomPosOne => random_pos_add(subject_f64, rng.gen_range(0..16), -1),
            MutatorFns::Truncate => subject_f64.trunc(),
            MutatorFns::RandTruncate => {
                let offset = rng.gen_range(1..=6) as f64;
                (subject_f64 * offset).trunc() / offset
            }
        };
        Subject(mutated_result.to_string())
    }

    fn fitness(subject: &Self::Subject) -> Fitness {
        subject.measure()
    }
}

#[derive(Debug, Copy, Clone, Default)]
enum ReproductionFns {
    #[default]
    NoOp,
    SexualGenetic,
    SexualHalf,
    ASexual,
    ZipDecimal,
}

impl ApplyReproduction for ReproductionFns {
    type Subject = Subject;

    fn apply(
        &self,
        _context: &GaContext,
        subject_a: &Self::Subject,
        subject_b: &Self::Subject,
    ) -> (Self::Subject, Self::Subject) {
        let a = subject_a.as_f64();
        let b = subject_b.as_f64();

        let (a, b) = match self {
            Self::NoOp => panic!("noop reproductive fn only used to satisfy default impl"),
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

fn random_f64(rng: &mut impl Rng) -> f64 {
    rng.gen::<f64>()
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

    let create_subject_fn =
        Box::new(|_context: &GaContext| -> Subject { random_f64(&mut rng::thread_rng()).into() });

    let ga_options = GeneticAlgorithmOptions {
        fitness_initial_to_target_range: 0f64..target_fitness,
        fitness_range: 0f64..target_fitness,
        create_subject_fn: create_subject_fn.clone(),
        actions: DefaultActions::<_, _, _, _, _, _, _, EmptyDedupe> {
            prune: PruneAction::new(PruneExtraBackSkipFirst::new(
                population_size - (population_size as f64 * 0.33).round() as usize,
            )),
            mutation: GenericMutator::new(ApplyMutationOptions {
                clone_on_mutation: false,
                overall_mutation_chance: 0.10,
                mutation_actions: WeightedActionsSampleOne(vec![
                    (MutatorFns::AddRandomPosOne, 0.75).into(),
                    (MutatorFns::SubRandomPosOne, 0.75).into(),
                    (MutatorFns::Truncate, 0.05).into(),
                    (MutatorFns::RandTruncate, 0.2).into(),
                    (MutatorFns::AddRandom, 0.75).into(),
                    (MutatorFns::SubRandom, 0.75).into(),
                    (MutatorFns::AddOne, 0.25).into(),
                    (MutatorFns::SubOne, 0.25).into(),
                ]),
            }),
            reproduction: GenericReproducer::new(ApplyReproductionOptions {
                selector: SelectRandomManyWithBias::new(population_size / 10, Bias::Front),
                overall_reproduction_chance: 1.0,
                reproduction_actions: WeightedActionsSampleOne(vec![
                    (ReproductionFns::SexualHalf, 0.50).into(),
                    (ReproductionFns::SexualGenetic, 0.75).into(),
                    (ReproductionFns::ASexual, 0.10).into(),
                    (ReproductionFns::ZipDecimal, 0.60).into(),
                ]),
            }),
            ..Default::default()
        },
    };

    let ga_runner_options = GaRunnerOptions {
        debug_print: Some(debug_print),
        before_each_generation: Some(|ga_iter_state| {
            if ga_iter_state.context().generation == 0 {
                return None;
            }
            if ga_iter_state.context().generation % 1000000 == 0 {
                debug!("generation: {}", ga_iter_state.context().generation);
            }
            None
        }),
        ..Default::default()
    };

    let population = create_population_pool(CreatePopulationOptions {
        population_size,
        create_subject_fn: create_subject_fn.clone(),
    });

    info!("starting generation loop");
    ga_runner(ga_options, ga_runner_options, population);
    info!("done")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest(initial, pos, amount, expected_len, expected_result,
        case::zero(0.0, 0, 0, 1, 0e-1),
        case::one(0.0, 1, 1, 3, 1e-1),
        case::five_five(0.5, 1, 5, 1, 1e-0),
        case::neg_one(0.0, 1, - 1, 3, - 1e-1),
        case::ten(0.0, 10, 1, 12, 1e-10),
        case::fifteen(0.0, 15, 1, 17, 1e-15),
        case::sixteen(0.0, 16, 1, 18, 1e-16),
        case::seventeen(0.0, 20, 1, 22, 1e-20),
        case::seventeen(0.0, 50, 1, 52, 1e-50),
    )]
    fn test_random_pos_add(
        initial: f64,
        pos: usize,
        amount: i8,
        expected_len: usize,
        expected_result: f64,
    ) {
        let result = random_pos_add(initial, pos, amount);
        assert_eq!(result, expected_result);
        assert_eq!(result.abs().to_string().len(), expected_len);
    }
}
