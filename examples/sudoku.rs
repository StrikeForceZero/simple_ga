use std::collections::HashSet;
use std::fmt::{Display, Formatter};

use rand::prelude::{IteratorRandom, SliceRandom};
use rand::Rng;
use tracing::{debug, info};

use simple_ga::ga::{
    create_population_pool, CreatePopulationOptions, GaContext, GeneticAlgorithmOptions,
};
use simple_ga::ga::fitness::{Fit, Fitness};
use simple_ga::ga::ga_runner::{ga_runner, GaRunnerOptions};
use simple_ga::ga::mutation::{ApplyMutation, ApplyMutationOptions};
use simple_ga::ga::reproduction::{ApplyReproduction, ApplyReproductionOptions};
use simple_ga::ga::subject::GaSubject;
use simple_ga::util::rng;

#[derive(Debug, Copy, Clone, PartialEq, Default)]
struct SudokuValidationGroup {
    correct: usize,
    wrong: usize,
    unknown: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Default)]
struct SudokuValidationResult {
    columns: SudokuValidationGroup,
    rows: SudokuValidationGroup,
    sub_grids: SudokuValidationGroup,
}

impl SudokuValidationResult {
    fn aggregate(&self) -> SudokuValidationGroup {
        SudokuValidationGroup {
            correct: self.columns.correct + self.rows.correct + self.sub_grids.correct,
            wrong: self.columns.wrong + self.rows.wrong + self.sub_grids.wrong,
            unknown: self.columns.unknown + self.rows.unknown + self.sub_grids.unknown,
        }
    }
    fn validate(board: BoardData) -> Self {
        let mut result = SudokuValidationResult::default();

        fn is_valid_group(group: &[u8; 9]) -> SudokuValidationGroup {
            let mut seen = HashSet::new();
            let mut validation_group = SudokuValidationGroup::default();

            for &num in group {
                if num == 0 {
                    validation_group.unknown += 1;
                } else if (1..=9).contains(&num) && seen.insert(num) {
                    validation_group.correct += 1;
                } else {
                    validation_group.wrong += 1;
                }
            }
            validation_group
        }

        // Check rows
        for row in &board {
            let validation_group = is_valid_group(row);
            result.rows.correct += validation_group.correct;
            result.rows.wrong += validation_group.wrong;
            result.rows.unknown += validation_group.unknown;
        }

        // Check columns
        for col in 0..9 {
            let mut column = [0; 9];
            for row in 0..9 {
                column[row] = board[row][col];
            }
            let validation_group = is_valid_group(&column);
            result.columns.correct += validation_group.correct;
            result.columns.wrong += validation_group.wrong;
            result.columns.unknown += validation_group.unknown;
        }

        // Check 3x3 subgrids
        for box_row in 0..3 {
            for box_col in 0..3 {
                let mut subgrid = [0; 9];
                for row in 0..3 {
                    for col in 0..3 {
                        subgrid[row * 3 + col] = board[box_row * 3 + row][box_col * 3 + col];
                    }
                }
                let validation_group = is_valid_group(&subgrid);
                result.sub_grids.correct += validation_group.correct;
                result.sub_grids.wrong += validation_group.wrong;
                result.sub_grids.unknown += validation_group.unknown;
            }
        }

        result
    }
}

fn fitness_numerator(sudoku_validation_result: SudokuValidationResult) -> f64 {
    let aggregate = sudoku_validation_result.aggregate();

    let unknown = aggregate.unknown as Fitness * UNKNOWN_WEIGHT;
    let wrong = aggregate.wrong as Fitness * WRONG_WEIGHT;

    unknown + wrong
}

impl From<SudokuValidationResult> for Fitness {
    fn from(value: SudokuValidationResult) -> Self {
        fitness_numerator(value)
    }
}

const TOTAL_SQUARES: usize = 81;
const AGGREGATE_TOTAL: usize = TOTAL_SQUARES * 3;
const UNKNOWN_WEIGHT: f64 = 1.0;
const WRONG_WEIGHT: f64 = 10.0;
const MAX_FITNESS: f64 = WRONG_WEIGHT * AGGREGATE_TOTAL as Fitness;
const INITIAL_FITNESS: f64 = UNKNOWN_WEIGHT * AGGREGATE_TOTAL as Fitness;

type BoardLikeGroup<T> = [T; 9];
type BoardLikeData<T> = [BoardLikeGroup<T>; 9];
type BoardData = BoardLikeData<u8>;

#[derive(Default, Clone, PartialEq, Eq, Hash)]
struct Board(BoardData);

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
enum BornFrom {
    #[default]
    Spawn,
    Reproduction,
    Mutation,
}

impl Board {
    #[cfg(test)]
    fn validate(&self) -> SudokuValidationResult {
        SudokuValidationResult::validate(self.0)
    }
    fn validation_display_string(&self) -> String {
        #[derive(Default, Debug)]
        struct Pos {
            row_ix: usize,
            col_ix: usize,
        }

        impl Pos {
            fn new(row_ix: usize, col_ix: usize) -> Self {
                Pos { row_ix, col_ix }
            }
        }

        #[derive(Default, Debug)]
        struct RemappedBoardCell {
            pos: Pos,
            cell: u8,
        }

        impl RemappedBoardCell {
            fn new(pos: Pos, cell: u8) -> Self {
                RemappedBoardCell { pos, cell }
            }
        }

        #[derive(Debug, Copy, Clone)]
        enum InvalidType {
            Row,
            Col,
            SubGrid,
        }

        #[derive(Default, Debug)]
        struct InvalidFlags {
            row: bool,
            col: bool,
            sub_grid: bool,
        }

        impl InvalidFlags {
            fn set(&mut self, invalid_type: InvalidType) {
                match invalid_type {
                    InvalidType::Row => self.row = true,
                    InvalidType::Col => self.col = true,
                    InvalidType::SubGrid => self.sub_grid = true,
                }
            }
        }

        let mut invalids = BoardLikeData::<InvalidFlags>::default();

        let mark_invalid_row_col = |invalids: &mut BoardLikeData<InvalidFlags>,
                                    group: BoardLikeGroup<RemappedBoardCell>,
                                    invalid_type: InvalidType| {
            let mut seen = HashSet::new();
            for RemappedBoardCell { pos, cell } in group.iter() {
                if (1..=9).contains(cell) && seen.insert(*cell) {
                    // skip
                } else {
                    invalids[pos.row_ix][pos.col_ix].set(invalid_type);
                }
            }
        };
        let mark_invalid_sub_grid =
            |invalids: &mut BoardLikeData<InvalidFlags>,
             group: BoardLikeGroup<RemappedBoardCell>,
             invalid_type: InvalidType| {
                let mut seen = HashSet::new();
                let mut invalid_positions = Vec::new();
                for RemappedBoardCell { pos, cell } in group.iter() {
                    if (1..=9).contains(cell) && seen.insert(*cell) {
                        // skip
                    } else {
                        invalid_positions.push((pos.row_ix, pos.col_ix));
                    }
                }
                for (row_ix, col_ix) in invalid_positions {
                    let box_row_start = (row_ix / 3) * 3;
                    let box_col_start = (col_ix / 3) * 3;
                    for row in 0..3 {
                        for col in 0..3 {
                            invalids[box_row_start + row][box_col_start + col].set(invalid_type);
                        }
                    }
                }
            };

        // Mark invalid sub grids
        for box_row in 0..3 {
            for box_col in 0..3 {
                let mut subgrid = Vec::new();
                for row in 0..3 {
                    for col in 0..3 {
                        let global_row = box_row * 3 + row;
                        let global_col = box_col * 3 + col;
                        subgrid.push(RemappedBoardCell::new(
                            Pos::new(global_row, global_col),
                            self.0[global_row][global_col],
                        ));
                    }
                }
                mark_invalid_sub_grid(
                    &mut invalids,
                    subgrid
                        .try_into()
                        .expect("failed to convert vec into [_;9]"),
                    InvalidType::SubGrid,
                );
            }
        }

        // Mark invalid rows
        for (row_ix, row) in self.0.iter().enumerate() {
            let mut remapped_row = BoardLikeGroup::<RemappedBoardCell>::default();
            for (col_ix, col) in row.iter().enumerate() {
                remapped_row[col_ix] = RemappedBoardCell::new(Pos::new(row_ix, col_ix), *col);
            }
            mark_invalid_row_col(&mut invalids, remapped_row, InvalidType::Row);
        }

        // Mark invalid columns
        for col in 0..9 {
            let mut column = BoardLikeGroup::<RemappedBoardCell>::default();
            for row in 0..9 {
                column[row] = RemappedBoardCell::new(Pos::new(row, col), self.0[row][col]);
            }
            mark_invalid_row_col(&mut invalids, column, InvalidType::Col)
        }

        let mut output = String::new();
        for row in invalids {
            for col in row {
                output.push_str(match (col.row, col.col, col.sub_grid) {
                    (false, false, false) => "[   ]",
                    (true, false, false) => "[R  ]",
                    (false, true, false) => "[ C ]",
                    (true, true, false) => "[RC ]",
                    (false, false, true) => "[  S]",
                    (true, false, true) => "[R S]",
                    (false, true, true) => "[ CS]",
                    (true, true, true) => "[RCS]",
                })
            }
            output.push_str("\n");
        }
        output
    }

    fn full_display_string(&self) -> String {
        let display_string = self.to_string();
        let validation_display_string = self.validation_display_string();
        let mut output = String::new();
        for (left, right) in display_string
            .lines()
            .zip(validation_display_string.lines())
        {
            output.push_str(&format!("{left}   {right}\n"))
        }
        output
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for row in self.0.iter() {
            for &col in row.iter() {
                if col > 0 {
                    write!(f, "[{col}]")?;
                } else {
                    write!(f, "[ ]")?;
                }
            }
            writeln!(f, "")?;
        }
        write!(f, "")
    }
}

impl From<Board> for BoardData {
    fn from(value: Board) -> Self {
        value.0
    }
}

impl<'a> From<&'a Board> for &'a BoardData {
    fn from(value: &'a Board) -> Self {
        &value.0
    }
}

impl Fit<Fitness> for Board {
    fn measure(&self) -> Fitness {
        SudokuValidationResult::validate(self.0).into()
    }
}

#[derive(Default, Clone, PartialEq, Eq, Hash)]
struct WrappedBoard {
    board: Board,
    generation_born: usize,
    born_from: BornFrom,
}

impl WrappedBoard {
    fn create_spawn(board: Board, generation: usize) -> Self {
        Self {
            board,
            generation_born: generation,
            born_from: BornFrom::Spawn,
        }
    }
    fn create_reproduction(board: Board, generation: usize) -> Self {
        Self {
            board,
            generation_born: generation,
            born_from: BornFrom::Reproduction,
        }
    }
    fn create_mutation(board: Board, generation: usize) -> Self {
        Self {
            board,
            generation_born: generation,
            born_from: BornFrom::Mutation,
        }
    }
}

impl GaSubject for WrappedBoard {}

impl Fit<Fitness> for WrappedBoard {
    fn measure(&self) -> Fitness {
        self.board.measure()
    }
}

enum MutatorFn {
    RotateRow,
    RandomFill,
    RandomOverwrite,
}

impl ApplyMutation for MutatorFn {
    type Subject = WrappedBoard;

    fn apply(&self, context: &GaContext, subject: &Self::Subject) -> Self::Subject {
        let rng = &mut rng::thread_rng();
        let mut subject = subject.board.clone();
        fn random_cell(
            rng: &mut impl Rng,
            subject: &Board,
            predicate: impl Fn(u8) -> bool,
        ) -> Option<(usize, usize, u8)> {
            let cells = subject
                .0
                .iter()
                .enumerate()
                .flat_map(|(row_ix, row)| {
                    row.iter()
                        .enumerate()
                        .map(move |(col_ix, col)| (row_ix, col_ix, *col))
                })
                .filter(|(_, _, col)| (&predicate)(*col));
            cells.choose(rng)
        }
        WrappedBoard::create_mutation(
            match self {
                Self::RotateRow => {
                    let Some(random_row) = subject.0.choose_mut(rng) else {
                        unreachable!();
                    };
                    if rng.gen_bool(0.5) {
                        random_row.rotate_left(1);
                    } else {
                        random_row.rotate_right(1);
                    }
                    subject
                }
                Self::RandomFill => {
                    let Some((row, col, _)) = random_cell(rng, &subject, |cell| cell == 0) else {
                        return WrappedBoard::create_mutation(subject, context.generation);
                    };
                    let Some(random_cell) = subject.0.get_mut(row).and_then(|row| row.get_mut(col))
                    else {
                        unreachable!();
                    };
                    *random_cell = rng.gen_range(1..=9);
                    subject
                }
                Self::RandomOverwrite => {
                    let Some((row, col, _)) = random_cell(rng, &subject, |cell| cell > 0) else {
                        return WrappedBoard::create_mutation(subject, context.generation);
                    };
                    let Some(random_cell) = subject.0.get_mut(row).and_then(|row| row.get_mut(col))
                    else {
                        unreachable!();
                    };
                    *random_cell = rng.gen_range(1..=9);
                    subject
                }
            },
            context.generation,
        )
    }

    fn fitness(subject: &Self::Subject) -> Fitness {
        subject.measure()
    }
}

enum ReproductionFn {
    RandomMix,
}

impl ApplyReproduction for ReproductionFn {
    type Subject = WrappedBoard;

    fn apply(
        &self,
        context: &GaContext,
        subject_a: &Self::Subject,
        subject_b: &Self::Subject,
    ) -> (Self::Subject, Self::Subject) {
        let rng = &mut rng::thread_rng();
        let subject_a = &subject_a.board;
        let subject_b = &subject_b.board;
        match self {
            Self::RandomMix => {
                let mut new_a = BoardData::default();
                let mut new_b = BoardData::default();
                for (row_ix, (a, b)) in subject_a.0.iter().zip(subject_b.0.iter()).enumerate() {
                    let (a, b) = a.iter().zip(b).enumerate().fold(
                        ([0u8; 9], [0u8; 9]),
                        |(mut a, mut b), (ix, (&a_val, &b_val))| {
                            let (new_a_val, new_b_val) = if rng.gen_bool(0.5) {
                                (a_val, b_val)
                            } else {
                                (b_val, a_val)
                            };
                            a[ix] = new_a_val;
                            b[ix] = new_b_val;
                            (a, b)
                        },
                    );
                    new_a[row_ix] = a;
                    new_b[row_ix] = b;
                }
                (
                    WrappedBoard::create_reproduction(Board(new_a), context.generation),
                    WrappedBoard::create_reproduction(Board(new_b), context.generation),
                )
            }
        }
    }

    fn fitness(subject: &Self::Subject) -> Fitness {
        subject.measure()
    }
}

fn main() {
    let population_size = 50;
    simple_ga_internal_lib::tracing::init_tracing();
    let target_fitness = 0.0;
    fn debug_print(subject: &WrappedBoard) {
        let fitness = subject.measure();
        debug!(
            "best: ({fitness}) generation born: {} via {:?}\n{}",
            subject.generation_born,
            subject.born_from,
            subject.board.full_display_string()
        );
    }

    let create_subject_fn = Box::new(|ga_context: &GaContext| {
        let mut wrapped_board = WrappedBoard::create_spawn(Board::default(), ga_context.generation);
        let board = &mut wrapped_board.board.0;
        let rng = &mut rng::thread_rng();
        if rng.gen_bool(0.1) {
            for row in board.iter_mut() {
                for col in row.iter_mut() {
                    *col = rng.gen_range(1..=9);
                }
            }
        } else if rng.gen_bool(0.75) {
            const FULL: BoardLikeGroup<u8> = [1, 2, 3, 4, 5, 6, 7, 8, 9];
            for row in board.iter_mut() {
                *row = FULL;
                if rng.gen_bool(0.5) {
                    row.rotate_left(rng.gen_range(0..9));
                } else {
                    row.rotate_left(rng.gen_range(0..9));
                }
            }
        }
        wrapped_board
    });

    let ga_options = GeneticAlgorithmOptions {
        remove_duplicates: true,
        fitness_initial_to_target_range: INITIAL_FITNESS..target_fitness,
        fitness_range: target_fitness..MAX_FITNESS,
        create_subject_fn: create_subject_fn.clone(),
        cull_amount: (population_size as f32 * 0.5).round() as usize,
        mutation_options: ApplyMutationOptions {
            clone_on_mutation: true,
            multi_mutation: false,
            overall_mutation_chance: 0.25,
            mutation_actions: vec![
                (MutatorFn::RandomFill, 0.10).into(),
                (MutatorFn::RotateRow, 0.25).into(),
                (MutatorFn::RandomOverwrite, 0.75).into(),
            ],
        },
        reproduction_options: ApplyReproductionOptions {
            reproduction_limit: (population_size as f32 * 0.25).round() as usize,
            multi_reproduction: false,
            overall_reproduction_chance: 0.25,
            reproduction_actions: vec![(ReproductionFn::RandomMix, 0.50).into()],
        },
    };

    let ga_runner_options = GaRunnerOptions {
        debug_print: Some(debug_print),
        log_on_mod_zero_for_generation_ix: 1000000,
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

    const CORRECT_BOARD: Board = Board([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ]);

    const SINGLE_WRONG: Board = Board([
        [3, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ]);

    const SINGLE_UNKNOWN: Board = Board([
        [0, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ]);

    #[test]
    fn test_validation() {
        assert_eq!(
            Board::default().validate().aggregate(),
            SudokuValidationGroup {
                unknown: TOTAL_SQUARES * 3,
                ..Default::default()
            }
        );

        assert_eq!(
            CORRECT_BOARD.validate().aggregate(),
            SudokuValidationGroup {
                correct: TOTAL_SQUARES * 3,
                ..Default::default()
            }
        );

        assert_eq!(
            SINGLE_WRONG.validate().aggregate(),
            SudokuValidationGroup {
                correct: TOTAL_SQUARES * 3 - 3,
                wrong: 3,
                ..Default::default()
            }
        );

        assert_eq!(
            SINGLE_UNKNOWN.validate().aggregate(),
            SudokuValidationGroup {
                correct: TOTAL_SQUARES * 3 - 3,
                wrong: 0,
                unknown: 1 * 3,
                ..Default::default()
            }
        );
    }

    #[test]
    fn test_fitness() {
        assert_eq!(Board::default().measure(), INITIAL_FITNESS);
        assert_eq!(CORRECT_BOARD.measure(), 0.0);
        assert_eq!(SINGLE_UNKNOWN.measure(), 3.0);
        assert_eq!(SINGLE_WRONG.measure(), 30.0);
    }
}
