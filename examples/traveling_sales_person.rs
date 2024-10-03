use itertools::Itertools;
use lazy_static::lazy_static;
use rand::prelude::SliceRandom;
use rand::Rng;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tracing::{debug, info};

use simple_ga::ga::action::DefaultActions;
use simple_ga::ga::dedupe::{DedupeAction, DefaultDedupe, EmptyDedupe};
use simple_ga::ga::fitness::{Fit, Fitness};
use simple_ga::ga::ga_iterator::GaIterState;
use simple_ga::ga::ga_runner::{ga_runner, GaRunnerCustomForEachGenerationResult, GaRunnerOptions};
use simple_ga::ga::inflate::InflateUntilFull;
use simple_ga::ga::mutation::{ApplyMutation, ApplyMutationOptions, GenericMutator};
use simple_ga::ga::prune::{DefaultPruneHalfBackSkipFirst, PruneAction};
use simple_ga::ga::reproduction::{
    ApplyReproduction, ApplyReproductionOptions, GenericReproducer, ReproductionResult,
};
use simple_ga::ga::select::SelectRandomManyWithBias;
use simple_ga::ga::subject::GaSubject;
use simple_ga::ga::{
    create_population_pool, CreatePopulationOptions, CreateSubjectFnArc, GaContext,
    GeneticAlgorithmOptions, WeightedActionsSampleOne,
};
use simple_ga::util::Bias;

trait SizeHintCollapse {
    fn collapse_max(&self) -> usize;
}

impl SizeHintCollapse for (usize, Option<usize>) {
    fn collapse_max(&self) -> usize {
        self.1.unwrap_or(self.0).max(self.0)
    }
}

#[derive(Debug, Clone)]
struct City {
    id: usize,
    x: f64,
    y: f64,
}

impl City {
    fn distance_to(&self, other: &City) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

impl Display for City {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id)
    }
}

impl PartialEq for City {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for City {}

impl Hash for City {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Display for Route {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]",
            self.cities.iter().map(|city| city.to_string()).join(", ")
        )
    }
}

#[derive(Default, Clone, Debug, PartialEq, Eq, Hash)]
struct Route {
    cities: Vec<&'static City>,
}

impl Route {
    fn total_distance(&self) -> f64 {
        let Some(first_city) = self.cities.first() else {
            panic!("expected at least 2 cities");
        };
        let Some(last_city) = self.cities.last() else {
            panic!("expected at least 2 cities");
        };
        self.cities
            .windows(2)
            .map(|pair| pair[0].distance_to(pair[1]))
            .sum::<f64>()
            + last_city.distance_to(first_city)
    }
    #[allow(dead_code)]
    fn calculate_fitness(&self) -> f64 {
        1.0 / self.total_distance()
    }
    #[allow(dead_code)]
    fn is_best_possible_route(&self) -> bool {
        self.is_best_possible_route_full().0
    }
    fn is_best_possible_route_full(&self) -> (bool, Option<Route>) {
        let num_cities = self.cities.len();
        let given_route_distance = self.total_distance();

        // Generate all possible routes
        let all_routes = self.cities.clone().into_iter().permutations(num_cities);
        let size = all_routes.size_hint();
        let mut ix: u128 = 0;
        // Check if the given route is the shortest
        for r in all_routes {
            if ix % 10000000 == 0 {
                let size = size.collapse_max();
                let percent = (ix as f64 / size as f64 * 10000.0).round() / 100.0;
                debug!("checking {ix}/{size} {percent}%");
            }
            let current_route = Route { cities: r };
            if current_route.total_distance() < given_route_distance {
                debug!(
                    "shorter: {current_route} - {}",
                    current_route.total_distance()
                );
                return (false, Some(current_route)); // Found a shorter route
            }
            ix += 1;
        }
        (true, None) // Given route is the shortest
    }
    #[allow(dead_code)]
    fn shortest(&self) -> Route {
        let num_cities = self.cities.len();
        let all_routes = self.cities.clone().into_iter().permutations(num_cities);
        let mut shortest_distance = self.total_distance();
        let mut shortest_route = self.clone();
        for r in all_routes {
            let current_route = Route { cities: r };
            let distance = current_route.total_distance();
            if distance < shortest_distance {
                shortest_distance = distance;
                shortest_route = current_route;
            }
        }
        shortest_route
    }
}

impl GaSubject for Route {}

impl Fit<Fitness> for Route {
    fn measure(&self) -> Fitness {
        self.calculate_fitness()
    }
}

fn generate_cities(num_cities: usize, width: f64, height: f64) -> Vec<City> {
    let mut rng = rand::thread_rng();
    (0..num_cities)
        .map(|id| City {
            id,
            x: rng.gen_range(0.0..width),
            y: rng.gen_range(0.0..height),
        })
        .collect()
}

#[derive(Debug, Copy, Clone, Default)]
enum Mutation {
    #[default]
    Swap,
}

impl<Data> ApplyMutation<Data> for Mutation
where
    Data: Default,
{
    type Subject = Route;

    fn apply(&self, _context: &GaContext<Data>, subject: &Self::Subject) -> Self::Subject {
        let rng = &mut simple_ga::util::rng::thread_rng();
        let mut subject = subject.clone();
        match self {
            Self::Swap => loop {
                let i = rng.gen_range(0..subject.cities.len());
                let j = rng.gen_range(0..subject.cities.len());
                if i == j {
                    continue;
                }
                subject.cities.swap(i, j);
                break;
            },
        }
        subject
    }

    fn fitness(subject: &Self::Subject) -> Fitness {
        subject.calculate_fitness()
    }
}

#[derive(Debug, Copy, Clone, Default)]
enum Reproduction {
    #[default]
    Reproduce,
}

impl<Data> ApplyReproduction<Data> for Reproduction
where
    Data: Default,
{
    type Subject = Route;

    fn apply(
        &self,
        _context: &GaContext<Data>,
        subject_a: &Self::Subject,
        subject_b: &Self::Subject,
    ) -> Option<ReproductionResult<Self::Subject>> {
        let mut rng = &mut simple_ga::util::rng::thread_rng();
        match self {
            Reproduction::Reproduce => {
                let size = subject_a.cities.len();
                let (start, end) = {
                    let mut indices = (0..size).collect::<Vec<_>>();
                    indices.shuffle(&mut rng);
                    (indices[0], indices[1])
                };
                let (start, end) = (start.min(end), start.max(end));

                let mut child_cities = vec![None; size];
                for i in start..end {
                    child_cities[i] = Some(subject_a.cities[i]);
                }

                let mut pointer = 0;
                for &city in &subject_b.cities {
                    if !child_cities.contains(&Some(city)) {
                        while child_cities[pointer].is_some() {
                            pointer += 1;
                        }
                        child_cities[pointer] = Some(city);
                    }
                }

                let route = Route {
                    cities: child_cities.into_iter().flatten().collect(),
                };
                Some(ReproductionResult::Single(route))
            }
        }
    }

    fn fitness(subject: &Self::Subject) -> Fitness {
        subject.measure()
    }
}

const TARGET_FITNESS: Fitness = 1.0;
const INITIAL_FITNESS: Fitness = 0.0;
const MAX_FITNESS: Fitness = Fitness::MAX;
const MIN_FITNESS: Fitness = 0.0;
const NUM_CITIES: usize = 12;

#[derive(Debug, Clone, Default)]
struct EmptyData;

fn main() {
    let population_size = 1000;
    simple_ga_internal_lib::tracing::init_tracing();

    lazy_static! {
        static ref CITIES: Vec<City> = generate_cities(NUM_CITIES, 100.0, 100.0);
    }
    let shuffled_cities = || {
        CITIES
            .choose_multiple(&mut simple_ga::util::rng::thread_rng(), CITIES.len())
            .collect()
    };

    for city in CITIES.iter() {
        println!("{city:?}");
    }

    let create_subject_fn: CreateSubjectFnArc<Route, EmptyData> =
        Arc::new(move |_ga_context: &GaContext<EmptyData>| Route {
            cities: shuffled_cities(),
        });

    fn debug_print(subject: &Route) {
        let fitness = subject.measure();
        let total_distance = subject.total_distance();
        debug!("best: (fit: {fitness}, dist: {total_distance}):\n{subject}");
    }

    fn check_if_best(
        iter_state: &mut GaIterState<Route, EmptyData>,
    ) -> Option<GaRunnerCustomForEachGenerationResult> {
        if iter_state.context().generation() == 0 {
            return None;
        }
        if iter_state.context().generation() % 100000 != 0 {
            return None;
        }
        info!("generation: {}", iter_state.context().generation());
        let best_subject = iter_state.population.subjects.first()?;
        debug!("checking if best subject is the best possible route...");
        let (is_best_route, better_route_opt) =
            best_subject.subject().is_best_possible_route_full();
        debug!("is best route: {is_best_route}");
        if is_best_route {
            debug!("{best_subject}");
            return Some(GaRunnerCustomForEachGenerationResult::Terminate);
        } else if let Some(better_route) = better_route_opt {
            // this is basically cheating but it's a good example of manipulating the runner
            iter_state.population.add(better_route.into());
        }
        None
    }

    let ga_options = GeneticAlgorithmOptions {
        fitness_initial_to_target_range: INITIAL_FITNESS..TARGET_FITNESS,
        fitness_range: MIN_FITNESS..MAX_FITNESS,
        actions: DefaultActions {
            prune: PruneAction::new(DefaultPruneHalfBackSkipFirst),
            mutation: GenericMutator::new(ApplyMutationOptions {
                clone_on_mutation: true,
                overall_mutation_chance: 0.75,
                mutation_actions: WeightedActionsSampleOne(vec![(Mutation::Swap, 0.5).into()]),
            }),
            reproduction: GenericReproducer::new(ApplyReproductionOptions {
                selector: SelectRandomManyWithBias::new(population_size / 4, Bias::Front),
                overall_reproduction_chance: 0.25,
                reproduction_actions: WeightedActionsSampleOne(vec![(
                    Reproduction::Reproduce,
                    0.50,
                )
                    .into()]),
            }),
            dedupe: DedupeAction::<_, EmptyDedupe>::default(),
            inflate: InflateUntilFull(create_subject_fn.clone()),
        },
        initial_data: EmptyData,
    };

    let ga_runner_options = GaRunnerOptions {
        debug_print: Some(debug_print),
        before_each_generation: Some(check_if_best),
        ..Default::default()
    };

    let population = create_population_pool(CreatePopulationOptions {
        population_size,
        create_subject_fn,
    });

    info!("starting generation loop");
    ga_runner(ga_options, ga_runner_options, population);
    info!("done")
}
