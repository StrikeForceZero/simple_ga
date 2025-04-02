use rand::prelude::{IteratorRandom, SliceRandom};
use rand::{thread_rng, Rng};
use serde::Serialize;
use simple_ga::ga::action::DefaultActions;
use simple_ga::ga::dedupe::{DedupeAction, DefaultDedupe, EmptyDedupe};
use simple_ga::ga::fitness::{Fit, Fitness};
use simple_ga::ga::ga_runner::{ga_runner, GaRunnerOptions};
use simple_ga::ga::inflate::InflateUntilFull;
use simple_ga::ga::mutation::{ApplyMutation, ApplyMutationOptions, GenericMutator};
use simple_ga::ga::prune::{PruneAction, PruneExtraBackSkipFirst};
use simple_ga::ga::reproduction::{
    ApplyReproduction, ApplyReproductionOptions, GenericReproducer, ReproductionResult,
};
use simple_ga::ga::select::SelectRandomManyWithBias;
use simple_ga::ga::subject::{GaSubject, Subject};
use simple_ga::ga::{
    create_population_pool, CreatePopulationOptions, GaContext, GeneticAlgorithmOptions,
    WeightedActionsSampleOne,
};
use simple_ga::util::{ApplyRatioFloat64, Bias};
use simple_ga_internal_lib::tracing::init_tracing;
use smallvec::{smallvec, SmallVec};
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Range;
use tracing::log::{debug, warn};

trait FloatSaturatingSub {
    type Output;
    type Other;
    fn saturating_sub(self, rhs: Self::Other) -> Self::Output;
}

impl FloatSaturatingSub for f64 {
    type Output = f64;
    type Other = f64;
    fn saturating_sub(self, rhs: Self::Other) -> Self::Output {
        saturating_sub(self, rhs, f64::MIN)
    }
}

fn saturating_sub(a: f64, b: f64, lower_bound: f64) -> f64 {
    let result = a - b;
    if result < lower_bound {
        lower_bound
    } else {
        result
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, serde::Serialize)]
#[repr(transparent)]
struct SmallSet<A>(SmallVec<A>)
where
    A: smallvec::Array,
    A::Item: Debug + Clone + PartialEq + Eq + Hash + serde::Serialize;

impl<A> SmallSet<A>
where
    A: smallvec::Array,
    A::Item: Debug + Clone + PartialEq + Eq + Hash + serde::Serialize,
{
    fn new() -> Self {
        SmallSet(SmallVec::new())
    }

    fn insert(&mut self, value: A::Item) -> bool {
        if self.0.contains(&value) {
            false
        } else {
            self.0.push(value);
            true
        }
    }

    fn remove(&mut self, value: &A::Item) -> bool {
        self.0.retain(|item| item != value);
        true
    }

    fn contains(&self, value: &A::Item) -> bool {
        self.0.contains(value)
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn into_inner(self) -> SmallVec<A> {
        self.0
    }

    fn inner(&self) -> &SmallVec<A> {
        &self.0
    }

    fn inner_mut(&mut self) -> &mut SmallVec<A> {
        &mut self.0
    }

    fn into_iter(self) -> impl IntoIterator<Item = A::Item> {
        self.0.into_iter()
    }

    fn iter(&self) -> impl Iterator<Item = &A::Item> {
        self.0.iter()
    }
}

impl<A> From<SmallVec<A>> for SmallSet<A>
where
    A: smallvec::Array,
    A::Item: Debug + Clone + PartialEq + Eq + Hash + serde::Serialize,
{
    fn from(value: SmallVec<A>) -> Self {
        SmallSet(value)
    }
}

impl<A> From<A> for SmallSet<A>
where
    A: smallvec::Array,
    A::Item: Debug + Clone + PartialEq + Eq + Hash + serde::Serialize,
{
    fn from(value: A) -> Self {
        SmallSet(SmallVec::from(value))
    }
}

const ROLES: [Role; 3] = [Role::Tank, Role::Damage, Role::Healer];

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
enum Pref {
    Willing = 1,
    Fallback = 2,
    Preferred = 3,
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash, serde::Serialize)]
struct Preferences {
    tank: Option<Pref>,
    damage: Option<Pref>,
    healer: Option<Pref>,
}

impl Preferences {
    fn get_pref_by_role(self, role: Role) -> Option<Pref> {
        match role {
            Role::Tank => self.tank,
            Role::Damage => self.damage,
            Role::Healer => self.healer,
        }
    }
    fn highest_pref(&self) -> Option<Pref> {
        [self.tank, self.damage, self.healer]
            .iter()
            .flatten()
            .copied()
            .max()
    }
    fn has_preferred(&self) -> bool {
        matches!(self.tank, Some(Pref::Preferred))
            || matches!(self.damage, Some(Pref::Preferred))
            || matches!(self.healer, Some(Pref::Preferred))
    }
    fn has_fallback(&self) -> bool {
        matches!(self.tank, Some(Pref::Fallback))
            || matches!(self.damage, Some(Pref::Fallback))
            || matches!(self.healer, Some(Pref::Fallback))
    }
    fn has_willing(&self) -> bool {
        matches!(self.tank, Some(Pref::Willing))
            || matches!(self.damage, Some(Pref::Willing))
            || matches!(self.healer, Some(Pref::Willing))
    }
    fn random_role(&self, rng: &mut impl Rng) -> Role {
        let mut roles = SmallSet::from(ROLES);
        if self.tank.is_none() {
            roles.remove(&Role::Tank);
        }
        if self.healer.is_none() {
            roles.remove(&Role::Healer);
        }
        if self.damage.is_none() {
            roles.remove(&Role::Damage);
        }
        if roles.is_empty() {
            panic!("no role");
        }
        *roles.into_inner().choose(rng).unwrap()
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash, serde::Serialize)]
struct Participant<Id> {
    id: Id,
    pref: Preferences,
}

impl<Id> Participant<Id>
where
    Id: Copy + Default,
{
    fn new(id: Id) -> Self {
        Self {
            id,
            ..Default::default()
        }
    }
    fn tank(mut self, pref: Pref) -> Self {
        self.pref.tank = Some(pref);
        self
    }
    fn damage(mut self, pref: Pref) -> Self {
        self.pref.damage = Some(pref);
        self
    }
    fn healer(mut self, pref: Pref) -> Self {
        self.pref.healer = Some(pref);
        self
    }
    fn prefer_tank(self) -> Self {
        self.tank(Pref::Preferred)
    }
    fn prefer_damage(self) -> Self {
        self.damage(Pref::Preferred)
    }
    fn prefer_healer(self) -> Self {
        self.healer(Pref::Preferred)
    }
    fn fallback_tank(self) -> Self {
        self.tank(Pref::Fallback)
    }
    fn fallback_damage(self) -> Self {
        self.damage(Pref::Fallback)
    }
    fn fallback_healer(self) -> Self {
        self.healer(Pref::Fallback)
    }
    fn willing_tank(self) -> Self {
        self.tank(Pref::Willing)
    }
    fn willing_damage(self) -> Self {
        self.damage(Pref::Willing)
    }
    fn willing_healer(self) -> Self {
        self.healer(Pref::Willing)
    }
    fn get_pref(&self, role: Role) -> Option<Pref> {
        match role {
            Role::Tank => self.pref.tank,
            Role::Damage => self.pref.damage,
            Role::Healer => self.pref.healer,
        }
    }
}

const MAX_TANK_COUNT: usize = 10;
const MAX_HEALER_COUNT: usize = 10;
const MAX_DAMAGE_COUNT: usize = 30;

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, serde::Serialize)]
struct Team<Id>
where
    Id: Debug + Clone + PartialEq + Eq + Hash + serde::Serialize,
{
    tanks: SmallSet<[Participant<Id>; MAX_TANK_COUNT]>,
    healers: SmallSet<[Participant<Id>; MAX_HEALER_COUNT]>,
    damage: SmallSet<[Participant<Id>; MAX_DAMAGE_COUNT]>,
}

impl<Id> Fit<Fitness> for Team<Id>
where
    Id: Debug + Default + Copy + Clone + PartialEq + Eq + Hash + serde::Serialize,
{
    fn measure(&self) -> Fitness {
        if self.tanks.is_empty() || self.damage.is_empty() || self.healers.is_empty() {
            return Fitness::MIN;
        }

        let tank_count = self.tanks.len();
        let healer_count = self.healers.len();
        let damage_count = self.damage.len();

        let mut fitness = 0_f64;

        // 10_000-point penalty for the healer-count to be unequal to the tank-count.
        if tank_count != healer_count {
            fitness = fitness.saturating_sub(10_000.0);
        }
        // 100_000_000-point penalty for damage-count to be > 3x the higher of tank-count or healer-count.
        if damage_count > healer_count * 3 {
            fitness = fitness.saturating_sub(100_000_000.0);
        }
        // 1_000-point penalty for the damage-count to be < 3x the healer-count or tank-count
        if damage_count < healer_count * 3 {
            fitness = fitness.saturating_sub(1_000.0);
        }

        // return none stops checking fitness and assumes invalid
        fn pref_fitness_penalty(role: Role, p_preferences: Preferences) -> Option<Fitness> {
            Some(match p_preferences.get_pref_by_role(role) {
                None => return None,
                Some(pref) => {
                    //  100-point penalty for a participant that says "preferred" and "willing" to be put into "willing".
                    if pref == Pref::Willing && p_preferences.has_preferred() {
                        if !p_preferences.has_willing() {
                            warn!(
                                "doesn't have willing - ruleset originally said 'and has willing'"
                            );
                            return None;
                        }
                        100.0
                    }
                    // 5-point penalty for a participant that says "preferred" to be put into "fallback".
                    else if pref == Pref::Fallback && p_preferences.has_preferred() {
                        5.0
                    }
                    // 15-point penalty for a participant that says "fallback" (but not "preferred") to be put into "willing".
                    else if pref == Pref::Willing
                        && p_preferences.has_fallback()
                        && !p_preferences.has_preferred()
                    {
                        15.0
                    } else {
                        0.0
                    }
                }
            })
        };

        for tank in self.tanks.iter() {
            let Some(penalty) = pref_fitness_penalty(Role::Tank, tank.pref) else {
                return Fitness::MIN;
            };
            fitness = fitness.saturating_sub(penalty);
        }
        for healer in self.healers.iter() {
            let Some(penalty) = pref_fitness_penalty(Role::Healer, healer.pref) else {
                return Fitness::MIN;
            };
            fitness = fitness.saturating_sub(penalty);
        }
        for damage in self.damage.iter() {
            let Some(penalty) = pref_fitness_penalty(Role::Damage, damage.pref) else {
                return Fitness::MIN;
            };
            fitness = fitness.saturating_sub(penalty);
        }

        fitness
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize)]
enum Role {
    Tank,
    Damage,
    Healer,
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum RoleResult<'a, T>
where
    T: Debug + Clone + PartialEq + Eq + Hash + serde::Serialize,
{
    Tank(&'a SmallSet<[T; MAX_TANK_COUNT]>),
    Damage(&'a SmallSet<[T; MAX_DAMAGE_COUNT]>),
    Healer(&'a SmallSet<[T; MAX_HEALER_COUNT]>),
}

#[derive(Debug, PartialEq)]
enum RoleResultMut<'a, T>
where
    T: Debug + Clone + PartialEq + Eq + Hash + serde::Serialize,
{
    Tank(&'a mut SmallSet<[T; MAX_TANK_COUNT]>),
    Damage(&'a mut SmallSet<[T; MAX_DAMAGE_COUNT]>),
    Healer(&'a mut SmallSet<[T; MAX_HEALER_COUNT]>),
}

impl<Id> Team<Id>
where
    Id: Debug + Clone + PartialEq + Eq + Hash + serde::Serialize,
{
    fn get_roles_by_role(&self, role: Role) -> RoleResult<Participant<Id>> {
        match role {
            Role::Tank => RoleResult::Tank(&self.tanks),
            Role::Damage => RoleResult::Damage(&self.damage),
            Role::Healer => RoleResult::Healer(&self.healers),
        }
    }
    fn get_roles_by_role_mut(&mut self, role: Role) -> RoleResultMut<Participant<Id>> {
        match role {
            Role::Tank => RoleResultMut::Tank(&mut self.tanks),
            Role::Damage => RoleResultMut::Damage(&mut self.damage),
            Role::Healer => RoleResultMut::Healer(&mut self.healers),
        }
    }
    fn assign_role(&mut self, participant: Participant<Id>, role: Role) {
        match role {
            Role::Tank => {
                self.damage.remove(&participant);
                self.healers.remove(&participant);
                self.tanks.insert(participant);
            }
            Role::Damage => {
                self.tanks.remove(&participant);
                self.healers.remove(&participant);
                self.damage.insert(participant);
            }
            Role::Healer => {
                self.tanks.remove(&participant);
                self.damage.remove(&participant);
                self.healers.insert(participant);
            }
        }
    }
}

impl<Id> GaSubject for Team<Id> where Id: Debug + Clone + PartialEq + Eq + Hash + serde::Serialize {}

fn main() {
    type Id = i32;
    type Data = Vec<Participant<Id>>;

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    enum MutatorFns {
        SwapRoles,
    }

    impl ApplyMutation for MutatorFns {
        type Subject = Team<Id>;

        fn apply(&self, _context: &GaContext, subject: &Self::Subject) -> Self::Subject {
            let rng = &mut thread_rng();
            let mut subject = subject.clone();
            let mut roles = SmallVec::from(ROLES);

            let from_role_index = rng.gen_range(0..roles.len());
            let from_role = roles.remove(from_role_index);

            let to_role_index = rng.gen_range(0..roles.len());
            let to_role = roles.remove(to_role_index);

            struct RemoveRandomParticipantOpt<'a, A, RNG>
            where
                A: smallvec::Array,
                A::Item: Debug + Clone + PartialEq + Eq + Hash + serde::Serialize,
                RNG: Rng,
            {
                participants: &'a mut SmallSet<A>,
                role_filter: Role,
                rng: RNG,
            }
            trait GetPref {
                type Output;
                fn get_pref(&self, role: Role) -> Self::Output;
                fn has_pref(&self, role: Role) -> bool;
            }
            impl GetPref for Participant<Id> {
                type Output = Option<Pref>;
                fn get_pref(&self, role: Role) -> Self::Output {
                    Participant::get_pref(self, role)
                }
                fn has_pref(&self, role: Role) -> bool {
                    self.get_pref(role).is_some()
                }
            }
            fn remove_random_participant<A>(
                mut options: RemoveRandomParticipantOpt<A, impl Rng>,
            ) -> Option<<A as smallvec::Array>::Item>
            where
                A: smallvec::Array,
                A::Item: Debug + Clone + PartialEq + Eq + Hash + GetPref + serde::Serialize,
            {
                let remove_index = options
                    .participants
                    .inner()
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.has_pref(options.role_filter))
                    .map(|(ix, _)| ix)
                    .choose(&mut options.rng)?;
                Some(options.participants.inner_mut().remove(remove_index))
            }
            let removed = match subject.get_roles_by_role_mut(from_role) {
                RoleResultMut::Tank(participants) => {
                    remove_random_participant(RemoveRandomParticipantOpt {
                        participants,
                        role_filter: to_role,
                        rng: rng.clone(),
                    })
                }
                RoleResultMut::Damage(participants) => {
                    remove_random_participant(RemoveRandomParticipantOpt {
                        participants,
                        role_filter: to_role,
                        rng: rng.clone(),
                    })
                }
                RoleResultMut::Healer(participants) => {
                    remove_random_participant(RemoveRandomParticipantOpt {
                        participants,
                        role_filter: to_role,
                        rng: rng.clone(),
                    })
                }
            };
            if let Some(removed) = removed {
                subject.assign_role(removed, to_role);
            }
            subject
        }

        fn fitness(subject: &Self::Subject) -> Fitness {
            subject.measure()
        }
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    enum ReproductionFns {
        MergeRole,
    }

    impl ApplyReproduction for ReproductionFns {
        type Subject = Team<Id>;

        fn apply(
            &self,
            _context: &GaContext,
            subject_a: &Self::Subject,
            subject_b: &Self::Subject,
        ) -> Option<ReproductionResult<Self::Subject>> {
            match self {
                ReproductionFns::MergeRole => {
                    let mut rng = thread_rng();
                    let role = ROLES.choose(&mut rng).unwrap();
                    let (new_a, new_b) = (subject_a.clone(), subject_b.clone());
                    fn merge<'a>(
                        mut new_a: Team<Id>,
                        mut new_b: Team<Id>,
                        a_iter: impl Iterator<Item = &'a Participant<Id>>,
                        b_iter: impl Iterator<Item = &'a Participant<Id>>,
                        role: Role,
                    ) -> (Team<Id>, Team<Id>) {
                        for &p in a_iter {
                            new_b.assign_role(p, role);
                        }
                        for &p in b_iter {
                            new_a.assign_role(p, role);
                        }
                        (new_a, new_b)
                    }
                    let (new_a, new_b) = match role {
                        Role::Tank => merge(
                            new_a,
                            new_b,
                            subject_a.tanks.iter(),
                            subject_b.tanks.iter(),
                            Role::Tank,
                        ),
                        Role::Damage => merge(
                            new_a,
                            new_b,
                            subject_a.damage.iter(),
                            subject_b.damage.iter(),
                            Role::Damage,
                        ),
                        Role::Healer => merge(
                            new_a,
                            new_b,
                            subject_a.healers.iter(),
                            subject_b.healers.iter(),
                            Role::Healer,
                        ),
                    };
                    Some(ReproductionResult::Double(new_a, new_b))
                }
            }
        }

        fn fitness(subject: &Self::Subject) -> Fitness {
            subject.measure()
        }
    }

    fn p(ids: &mut Range<Id>) -> Participant<Id> {
        Participant::new(ids.next().unwrap())
    }
    let mut ids: Range<Id> = 1..99;
    let ids = &mut ids;
    let mut data: Vec<Participant<Id>> = vec![
        p(ids).prefer_tank(),
        p(ids).prefer_damage(),
        p(ids).prefer_healer(),
        p(ids).prefer_tank().fallback_damage(),
        p(ids).prefer_tank().fallback_healer(),
        p(ids).prefer_damage().fallback_tank(),
        p(ids).prefer_damage().fallback_healer(),
        p(ids).willing_tank().willing_damage().willing_healer(),
        p(ids).willing_tank().willing_damage().willing_healer(),
        p(ids).prefer_tank().willing_damage().willing_healer(),
        p(ids).prefer_damage().willing_tank().willing_healer(),
        p(ids).prefer_damage().willing_tank().willing_healer(),
        p(ids).prefer_damage().willing_tank().willing_healer(),
        p(ids).prefer_damage().willing_tank().willing_healer(),
        p(ids).prefer_damage().willing_tank().willing_healer(),
        p(ids).prefer_damage().willing_tank().willing_healer(),
        p(ids).prefer_damage().willing_tank().willing_healer(),
        p(ids).prefer_damage().willing_tank().willing_healer(),
        p(ids).prefer_damage().willing_tank().willing_healer(),
        p(ids).prefer_damage().willing_tank().willing_healer(),
        p(ids).prefer_damage().willing_tank().willing_healer(),
        p(ids).prefer_healer().willing_tank().willing_damage(),
    ];

    fn create_subject_fn(_context: &GaContext, data: &mut Data) -> Team<Id> {
        let mut team = Team::default();
        let mut rng = thread_rng();
        for &p in data.iter() {
            // skip anyone without at least one preference
            if p.pref.highest_pref().is_none() {
                continue;
            }
            let random_role = p.pref.random_role(&mut rng);
            team.assign_role(p, random_role);
        }
        team
    }

    let population_size = 1000;
    let population = create_population_pool(
        CreatePopulationOptions {
            population_size,
            create_subject_fn,
        },
        &mut data,
    );

    fn debug_print(subject: &Team<Id>) {
        let fitness = subject.measure();
        println!("---");
        println!(
            "counts:\n\ttotal: {}\n\t\ttanks: {},\n\t\thealers: {},\n\t\tdamage: {}",
            subject.tanks.len() + subject.healers.len() + subject.damage.len(),
            subject.tanks.len(),
            subject.healers.len(),
            subject.damage.len()
        );
        println!(
            "healer to tank ratio: {}",
            subject.healers.len() as f64 / subject.tanks.len() as f64
        );
        println!(
            "damage to healer ratio: {}",
            subject.damage.len() as f64 / subject.healers.len() as f64
        );
        let serialized = serde_json::to_string(subject).unwrap();
        println!("best:\n\tfitness: {fitness}\n\tsubject: {serialized}");
    }

    let ga_options = GeneticAlgorithmOptions {
        fitness_initial_to_target_range: Fitness::MIN..0f64,
        fitness_range: Fitness::MIN..0f64,
        actions: DefaultActions {
            prune: PruneAction::new(PruneExtraBackSkipFirst::new(
                population_size.apply_ratio_round(0.33),
            )),
            mutation: GenericMutator::new(ApplyMutationOptions {
                clone_on_mutation: false,
                overall_mutation_chance: 0.10,
                mutation_actions: WeightedActionsSampleOne(vec![
                    (MutatorFns::SwapRoles, 0.75).into()
                ]),
            }),
            reproduction: GenericReproducer::new(ApplyReproductionOptions {
                selector: SelectRandomManyWithBias::new(population_size / 10, Bias::Front),
                overall_reproduction_chance: 1.0,
                reproduction_actions: WeightedActionsSampleOne(vec![(
                    ReproductionFns::MergeRole,
                    0.50,
                )
                    .into()]),
            }),
            dedupe: DedupeAction::<_, DefaultDedupe<_>>::default(),
            inflate: InflateUntilFull::new(create_subject_fn),
            // TODO: fix default
            // ..Default::default()
        },
        initial_data: Some(data),
    };

    let ga_runner_options = GaRunnerOptions {
        debug_print: Some(debug_print),
        before_each_generation: Some(
            |ga_iter_state: &mut simple_ga::ga::ga_iterator::GaIterState<Team<Id>, Data>| {
                if ga_iter_state.context().generation == 0 {
                    return None;
                }
                if ga_iter_state.context().generation % 10000 == 0 {
                    debug!("generation: {}", ga_iter_state.context().generation);
                }
                None
            },
        ),
        ..Default::default()
    };

    init_tracing();
    ga_runner(ga_options, ga_runner_options, population);
}
