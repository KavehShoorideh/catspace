//! LAYER 0, MODULE 6 (opposed choice), THE LINEAR PROGRAM OF THE MIXED CHOICE (plan 79.31 part B; SYSTEM_SPEC 11 "Layer 0 is made right", the block on belief and choosing by chance).
//! A small simplex written here over the exact numbers of `number` (no outside crate). It solves ONE thing: a table of numbers where one chooser picks a row, the other a column, both at once, the first wanting the number high and the second low.
//!
//! INTERFACE (caller: `choice::mixed_value` and `choice::mixed_table`; tests).
//!   `solve_matrix_game(&[Vec<Num>]) -> Result<MatrixGame, LpError>`; `MatrixGame { value, row, col }`: the value of the table, the best spread over the rows (a chance for each row, summing to one) and the best spread over the columns.
//!     The row spread guarantees at least `value` against every column; the column spread holds the row chooser to at most `value` against every row; both are tested by `MatrixGame::guarantees_meet`.
//!   `LpError::{Empty, Ragged, NotExact, Overflow}`: the table has no row or column, its rows differ in length, an entry is a real number (the program is exact and does not read reals), a number left the range of the exact numbers.
//!     An exact number that overflows is SAID (`Overflow`); there is no fall back to a floating answer.
//! METHOD: add a constant so every entry is at least one; then the column chooser's side is "maximise the sum of y subject to A y <= 1, y >= 0", a program that starts feasible at nought; the simplex with the smallest-index rule (no cycling) ends with the value 1/Z less
//! the constant, the column spread y / Z, and the row spread read off the objective row's slack entries over Z.
use crate::number::{Num, NumError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpError { Empty, Ragged, NotExact, Overflow }

impl std::fmt::Display for LpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            LpError::Empty => "a table has a row and a column",
            LpError::Ragged => "the rows of a table have one length",
            LpError::NotExact => "the program reads whole numbers and fractions, not reals",
            LpError::Overflow => "an exact number left the range of the exact numbers",
        })
    }
}

impl From<NumError> for LpError { fn from(_: NumError) -> LpError { LpError::Overflow } }

/// The value of a table of numbers and the best spread of each chooser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixGame { pub value: Num, pub row: Vec<Num>, pub col: Vec<Num> }

impl MatrixGame {
    /// The two guarantees meet: the row spread gets at least `value` against every column, the column spread gives at most `value` against every row, and each spread sums to one.
    pub fn guarantees_meet(&self, m: &[Vec<Num>]) -> bool {
        let sum = |v: &[Num]| v.iter().try_fold(Num::whole(0), |a, x| a.add(x)).ok();
        if sum(&self.row) != Some(Num::whole(1)) || sum(&self.col) != Some(Num::whole(1)) { return false; }
        if self.row.iter().chain(self.col.iter()).any(|p| *p < Num::whole(0)) { return false; }
        let (rows, cols) = (m.len(), m[0].len());
        let against_col = |j: usize| (0..rows).try_fold(Num::whole(0), |a, i| a.add(&self.row[i].mul(&m[i][j])?)).ok();
        let against_row = |i: usize| (0..cols).try_fold(Num::whole(0), |a, j| a.add(&self.col[j].mul(&m[i][j])?)).ok();
        (0..cols).all(|j| against_col(j).is_some_and(|v| v >= self.value)) && (0..rows).all(|i| against_row(i).is_some_and(|v| v <= self.value))
    }
}

pub fn solve_matrix_game(m: &[Vec<Num>]) -> Result<MatrixGame, LpError> {
    let rows = m.len();
    if rows == 0 || m[0].is_empty() { return Err(LpError::Empty); }
    let cols = m[0].len();
    if m.iter().any(|r| r.len() != cols) { return Err(LpError::Ragged); }
    if m.iter().flatten().any(|x| !x.is_exact()) { return Err(LpError::NotExact); }
    let least = m.iter().flatten().copied().min().expect("a table has an entry");
    let shift = Num::whole(1).sub(&least)?;                                                   // every entry plus the shift is at least one
    let (zero, one) = (Num::whole(0), Num::whole(1));
    // the tableau: rows 0..rows are the constraints (columns: y, slacks, right side); the last row is the objective z - sum y = 0
    let width = cols + rows + 1;
    let mut t: Vec<Vec<Num>> = Vec::with_capacity(rows + 1);
    for i in 0..rows {
        let mut line = vec![zero; width];
        for j in 0..cols { line[j] = m[i][j].add(&shift)?; }
        line[cols + i] = one;
        line[width - 1] = one;
        t.push(line);
    }
    let mut obj = vec![zero; width];
    for j in 0..cols { obj[j] = Num::whole(-1); }
    t.push(obj);
    let mut basis: Vec<usize> = (0..rows).map(|i| cols + i).collect();
    loop {
        // the entering variable: the smallest index with a negative objective entry
        let Some(e) = (0..width - 1).find(|&j| t[rows][j] < zero) else { break };
        // the leaving row: least ratio of the right side to a positive entry, the smallest basic index among ties
        let mut pick: Option<(usize, Num)> = None;
        for i in 0..rows {
            if t[i][e] > zero {
                let ratio = t[i][width - 1].div(&t[i][e])?;
                let better = match &pick { None => true, Some((p, r)) => ratio < *r || (ratio == *r && basis[i] < basis[*p]) };
                if better { pick = Some((i, ratio)); }
            }
        }
        let (l, _) = pick.expect("the program is bounded: every entry is positive");
        let piv = t[l][e];
        for k in 0..width { t[l][k] = t[l][k].div(&piv)?; }
        for i in 0..=rows {
            if i != l && t[i][e] != zero {
                let f = t[i][e];
                for k in 0..width { let d = f.mul(&t[l][k])?; t[i][k] = t[i][k].sub(&d)?; }
            }
        }
        basis[l] = e;
    }
    let z = t[rows][width - 1];
    let mut y = vec![zero; cols];
    for (i, &b) in basis.iter().enumerate() { if b < cols { y[b] = t[i][width - 1]; } }
    let col: Vec<Num> = y.iter().map(|v| v.div(&z)).collect::<Result<_, _>>()?;
    let row: Vec<Num> = (0..rows).map(|i| t[rows][cols + i].div(&z)).collect::<Result<_, _>>()?;
    let value = one.div(&z)?.sub(&shift)?;
    Ok(MatrixGame { value, row, col })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maker::Maker;

    fn n(v: i64) -> Num { Num::whole(v) }
    fn q(a: i64, b: i64) -> Num { Num::frac(a, b).unwrap() }
    fn table(rows: &[&[i64]]) -> Vec<Vec<Num>> { rows.iter().map(|r| r.iter().map(|x| n(*x)).collect()).collect() }

    #[test]
    fn matching_pennies_is_worth_nought_and_played_by_halves() {
        let g = solve_matrix_game(&table(&[&[1, -1], &[-1, 1]])).unwrap();
        assert_eq!(g.value, n(0));
        assert_eq!(g.row, vec![q(1, 2), q(1, 2)]);
        assert_eq!(g.col, vec![q(1, 2), q(1, 2)]);
    }

    #[test]
    fn known_small_tables_by_hand() {
        // a table with a pure answer: row 0 wins at once, with chance one
        let g = solve_matrix_game(&table(&[&[3, 2], &[1, 0]])).unwrap();
        assert_eq!((g.value, g.row.clone(), g.col.clone()), (n(2), vec![n(1), n(0)], vec![n(0), n(1)]));
        // the 2 by 2 table [[2, -1], [-1, 1]]: value (2 - 1) / (2 + 1 + 1 + 1) = 1/5, rows (2/5, 3/5), columns (2/5, 3/5)
        let g = solve_matrix_game(&table(&[&[2, -1], &[-1, 1]])).unwrap();
        assert_eq!(g.value, q(1, 5));
        assert_eq!(g.row, vec![q(2, 5), q(3, 5)]);
        assert_eq!(g.col, vec![q(2, 5), q(3, 5)]);
        // stone, paper, scissors: nought, a third each
        let g = solve_matrix_game(&table(&[&[0, -1, 1], &[1, 0, -1], &[-1, 1, 0]])).unwrap();
        assert_eq!((g.value, g.row.clone(), g.col.clone()), (n(0), vec![q(1, 3); 3], vec![q(1, 3); 3]));
        // a table with fractions: [[1/2, 3/4], [1/4, 1]] has the pure answer 1/2 at row 0? row 0 min is 1/2, row 1 min is 1/4; col 0 max 1/2, col 1 max 1 : saddle 1/2
        let g = solve_matrix_game(&[vec![q(1, 2), q(3, 4)], vec![q(1, 4), n(1)]]).unwrap();
        assert_eq!(g.value, q(1, 2));
        // a single entry
        assert_eq!(solve_matrix_game(&table(&[&[7]])).unwrap().value, n(7));
        // a row that is worse in every column is never played
        let g = solve_matrix_game(&table(&[&[4, 1], &[0, 0], &[1, 3]])).unwrap();
        assert_eq!(g.row[1], n(0));
    }

    #[test]
    fn a_table_that_cannot_be_read_is_refused() {
        assert_eq!(solve_matrix_game(&[]), Err(LpError::Empty));
        assert_eq!(solve_matrix_game(&[vec![]]), Err(LpError::Empty));
        assert_eq!(solve_matrix_game(&[vec![n(1), n(2)], vec![n(3)]]), Err(LpError::Ragged));
        assert_eq!(solve_matrix_game(&[vec![Num::real(0.1).unwrap()]]), Err(LpError::NotExact));
    }

    #[test]
    fn an_exact_number_that_leaves_the_range_is_said_and_never_made_a_real() {
        let big = n(i64::MAX / 2);
        assert_eq!(solve_matrix_game(&[vec![big, n(-1)], vec![n(-1), big]]), Err(LpError::Overflow));
    }

    /// The two guarantees meet on every table the maker makes: the row spread gets at least the value against every column, the column spread gives at most the value against every row.
    /// Mirrors Ideas.Layer0.Choice.guarantees_meet_of_certificate, certificate_value_unique (two such spreads give THE value). That a table has them is the minimax theorem, which is OPEN in Lean and is here tried on cases.
    #[test]
    fn the_guarantees_of_the_two_choosers_meet_on_made_tables() {
        let mut m = Maker::new(91);
        let mut mixed = 0;
        for _ in 0..1500 {
            let (r, c) = (1 + m.below(4) as usize, 1 + m.below(4) as usize);
            let t: Vec<Vec<Num>> = (0..r).map(|_| (0..c).map(|_| if m.chance() { n(m.range(-6, 6)) } else { q(m.range(-9, 9), m.range(1, 5)) }).collect()).collect();
            let g = solve_matrix_game(&t).unwrap();
            assert!(g.guarantees_meet(&t), "{t:?} -> {g:?}");
            let lower = (0..r).map(|i| t[i].iter().copied().min().unwrap()).max().unwrap();
            let upper = (0..c).map(|j| (0..r).map(|i| t[i][j]).max().unwrap()).min().unwrap();
            assert!(lower <= g.value && g.value <= upper, "the value lies between the lower and the upper of the pure choices");
            if lower < upper { mixed += 1; }
        }
        assert!(mixed > 300, "the maker makes tables without a pure answer ({mixed})");
    }
}
