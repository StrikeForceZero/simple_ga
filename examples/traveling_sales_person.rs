use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};

use itertools::Itertools;
use lazy_static::lazy_static;
use rand::prelude::SliceRandom;
use rand::Rng;
use tracing::{debug, info, warn};

use simple_ga::ga::{
    create_population_pool, CreatePopulationOptions, GaContext, GeneticAlgorithmOptions,
};
use simple_ga::ga::fitness::{Fit, Fitness};
use simple_ga::ga::ga_iterator::{GaIterator, GaIterOptions, GaIterState};
use simple_ga::ga::ga_runner::{ga_runner, GaRunnerOptions};
use simple_ga::ga::mutation::{ApplyMutation, ApplyMutationOptions};
use simple_ga::ga::reproduction::{ApplyReproduction, ApplyReproductionOptions};
use simple_ga::ga::subject::GaSubject;
use simple_ga::util::rng;

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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
    fn calculate_fitness(&self) -> f64 {
        1.0 / self.total_distance()
    }
    fn is_best_possible_route(&self) -> bool {
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
                return false; // Found a shorter route
            }
            ix += 1;
        }
        true // Given route is the shortest
    }
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

enum Mutation {
    Swap,
}

impl ApplyMutation for Mutation {
    type Subject = Route;

    fn apply(&self, _context: &GaContext, subject: &Self::Subject) -> Self::Subject {
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

enum Reproduction {
    Reproduce,
}

impl ApplyReproduction for Reproduction {
    type Subject = Route;

    fn apply(
        &self,
        _context: &GaContext,
        subject_a: &Self::Subject,
        subject_b: &Self::Subject,
    ) -> (Self::Subject, Self::Subject) {
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
                    cities: child_cities.into_iter().filter_map(|c| c).collect(),
                };
                (route.clone(), route)
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
    /*lazy_static! {
        static ref BEST_ROUTE: Route = Route {
            cities: CITIES
                .choose_multiple(&mut simple_ga::util::rng::thread_rng(), CITIES.len())
                .collect(),
        }
        .shortest();
        static ref BEST_ROUTE_DISTANCE: f64 = BEST_ROUTE.total_distance();
    }*/

    for city in CITIES.iter() {
        println!("{city:?}");
    }

    let create_subject_fn = Box::new(|ga_context: &GaContext| Route {
        cities: shuffled_cities(),
    });

    fn debug_print(subject: &Route) {
        let fitness = subject.measure();
        let total_distance = subject.total_distance();
        debug!("best: (fit: {fitness}, dist: {total_distance}):\n{subject}");
        /*
        let is_best_route = total_distance <= *BEST_ROUTE_DISTANCE;
        debug!("is best route: {is_best_route}");
        if !is_best_route {
            debug!("best: {} {}", *BEST_ROUTE, *BEST_ROUTE_DISTANCE);
        }
        */
    }

    fn check_if_best(_ga_context: &GaContext, subject: &Route) {
        debug!("checking if best subject is the best possible route...");
        let is_best_route = subject.is_best_possible_route();
        debug!("is best route: {is_best_route}");
        if is_best_route {
            debug!("{subject}");
            panic!("exiting");
        }
    }

    let ga_options = GeneticAlgorithmOptions {
        remove_duplicates: false,
        fitness_initial_to_target_range: INITIAL_FITNESS..TARGET_FITNESS,
        fitness_range: MIN_FITNESS..MAX_FITNESS,
        create_subject_fn: create_subject_fn.clone(),
        cull_amount: (population_size as f32 * 0.5).round() as usize,
        mutation_options: ApplyMutationOptions {
            clone_on_mutation: true,
            multi_mutation: false,
            overall_mutation_chance: 0.75,
            mutation_actions: vec![(Mutation::Swap, 0.5).into()],
        },
        reproduction_options: ApplyReproductionOptions {
            reproduction_limit: (population_size as f32 * 0.25).round() as usize,
            multi_reproduction: false,
            overall_reproduction_chance: 0.25,
            reproduction_actions: vec![(Reproduction::Reproduce, 0.50).into()],
        },
    };

    let ga_runner_options = GaRunnerOptions {
        debug_print: Some(debug_print),
        log_on_mod_zero_for_generation_ix: 100000,
        run_on_mod_zero_for_generation_ix: Some(check_if_best),
    };

    let population = create_population_pool(CreatePopulationOptions {
        population_size,
        create_subject_fn: create_subject_fn.clone(),
    });

    info!("starting generation loop");
    ga_runner(ga_options, ga_runner_options, population);
    info!("done")
}
