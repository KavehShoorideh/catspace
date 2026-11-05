//! THE VALUATION MINER (plan 76.4c's definition -- discovery/valuation_miner.py, its Python body -- with its loops here; plan 78.4): a game's units valued ONLY BY
//! OUTCOMES. THE ATOMS: per unit kind k, the count of the side to move's units of kind k less the other side's (read through `Game::units`); a kind whose
//! difference is 0 at every corpus position is CONSTANT (a royal unit: one each) -- it carries no outcome and takes no weight. THE MODEL: with m the weighted sum,
//! P(win) = sigma(s m - c), P(loss) = sigma(-s m - c), the draw the rest (a symmetric ordinal logistic, c >= 0 the draw band), fit by maximum likelihood over the
//! positions grouped by their atom vector. THE SEARCH, PRICED, INTEGERS FIRST: the continuous best weights; then, price by price (the sum of the weights' sizes),
//! the integer vectors nearest them -- each weight's SIGN the continuous fit's, its size the floor or ceiling of its share of the price (up to CHECK_PRICE every vector
//! of that sign pattern is read too) -- each fit for its own s and c; THE FIRST PRICE whose best vector's mean log loss is above the continuous fit's by no more than
//! ONE STANDARD ERROR of the paired difference, games as clusters, is the answer. Signed (78.4): a game whose goal is to shed its units values them below 0.
use crate::Record;
use cs_game::Game;
use std::collections::HashMap;

pub const CHECK_PRICE: u32 = 12;

/// The corpus: positions grouped by atom vector.
pub struct Corpus {
    pub kinds: Vec<String>,
    pub constant: Vec<String>,
    pub vec: Vec<Vec<f64>>,          // per group: the atom vector over `kinds` (the live ones)
    pub n: Vec<[f64; 3]>,            // per group: losses, draws, wins (the side to move's)
    pub pos_group: Vec<u32>,
    pub pos_game: Vec<u32>,
    pub pos_out: Vec<u8>,
    pub games: usize,
}

/// The corpus of `records` (finished ones), every position from ply `from_ply` on, read from the side to move; `make` builds a record's game (from its start).
pub fn corpus<G: Game, F: Fn(&Record) -> Option<G>>(make: F, records: &[Record], from_ply: usize) -> Corpus {
    let mut kind_ix: HashMap<String, usize> = HashMap::new();
    let mut rows: Vec<(u32, Vec<i32>, u8)> = vec![];
    let mut used = 0u32;
    for (gi, r) in records.iter().enumerate() {
        if r.outcome.is_empty() { continue; }
        let Some(g) = make(r) else { continue };
        let g = &g;
        used += 1;
        let mut s = g.initial();
        for (ply, name) in r.moves.iter().enumerate() {
            if ply >= from_ply {
                if let Some(me) = g.mover(&s) {
                    let mut v: Vec<i32> = vec![0; kind_ix.len()];
                    for (owner, kind) in g.units(&s) {
                        let n = kind_ix.len();
                        let k = *kind_ix.entry(kind).or_insert(n);
                        if k >= v.len() { v.resize(k + 1, 0); }
                        v[k] += if owner == me { 1 } else { -1 };
                    }
                    let best_other = r.outcome.iter().enumerate().filter(|(i, _)| *i != me).map(|(_, x)| *x).max().unwrap_or(0);
                    let o = if r.outcome[me] > best_other { 2 } else if r.outcome[me] < best_other { 0 } else { 1 };
                    rows.push((gi as u32, v, o));
                }
            }
            let Some(m) = g.find_move(&s, name) else { break };
            s = g.apply(&s, &m);
        }
    }
    let nk = kind_ix.len();
    let mut names = vec![String::new(); nk];
    for (k, i) in &kind_ix { names[*i] = k.clone(); }
    for r in rows.iter_mut() { r.1.resize(nk, 0); }
    let live: Vec<usize> = (0..nk).filter(|&k| rows.iter().any(|r| r.1[k] != 0)).collect();
    let mut order: Vec<usize> = live.clone();
    order.sort_by(|a, b| names[*a].cmp(&names[*b]));
    let mut gidx: HashMap<Vec<i32>, usize> = HashMap::new();
    let (mut vec, mut n, mut pos_group, mut pos_game, mut pos_out) = (vec![], vec![], vec![], vec![], vec![]);
    let mut game_ix: HashMap<u32, u32> = HashMap::new();
    for (gm, v, o) in &rows {
        let key: Vec<i32> = order.iter().map(|&k| v[k]).collect();
        let j = *gidx.entry(key.clone()).or_insert_with(|| { vec.push(key.iter().map(|x| *x as f64).collect()); n.push([0.0; 3]); vec.len() - 1 });
        n[j][*o as usize] += 1.0;
        pos_group.push(j as u32);
        let ng = game_ix.len() as u32;
        pos_game.push(*game_ix.entry(*gm).or_insert(ng));
        pos_out.push(*o);
    }
    let _ = used;
    Corpus { kinds: order.iter().map(|&k| names[k].clone()).collect(), constant: (0..nk).filter(|k| !live.contains(k)).map(|k| names[k].clone()).collect(),
             vec, n, pos_group, pos_game, pos_out, games: game_ix.len() }
}

fn sig(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) } }
fn softplus(u: f64) -> f64 { if u > 30.0 { u } else { (1.0 + u.exp()).ln() } }

fn probs(m: f64, c: f64) -> [f64; 3] {
    let pw = sig(m - c);
    let pl = sig(-m - c);
    [pl.max(1e-300), (1.0 - pw - pl).max(1e-300), pw.max(1e-300)]
}

/// The negative log likelihood of the groups at m_j and c, and its derivatives in m_j (per group) and in c.
fn nll_grad(m: &[f64], n: &[[f64; 3]], c: f64) -> (f64, Vec<f64>, f64) {
    let (mut l, mut gc) = (0.0, 0.0);
    let mut gm = vec![0.0; m.len()];
    for j in 0..m.len() {
        let pw = sig(m[j] - c);
        let pl = sig(-m[j] - c);
        let pd = (1.0 - pw - pl).max(1e-300);
        let [nl, nd, nw] = n[j];
        l -= nl * pl.max(1e-300).ln() + nd * pd.ln() + nw * pw.max(1e-300).ln();
        let dw = pw * (1.0 - pw);
        let dl = pl * (1.0 - pl);
        gm[j] = -(nw * (1.0 - pw) - nl * (1.0 - pl) + nd * (-dw + dl) / pd);
        gc += -(-nw * (1.0 - pw) - nl * (1.0 - pl) + nd * (dw + dl) / pd);
    }
    (l, gm, gc)
}

/// Maximum likelihood of weights `beta` over the columns of `x` (one row per group) and the band c = softplus(u): damped Newton on the analytic gradient with a
/// Hessian from its differences. -> (beta, c, nll).
pub fn fit(x: &[Vec<f64>], n: &[[f64; 3]], start: Option<(Vec<f64>, f64)>) -> (Vec<f64>, f64, f64) {
    let k = x.first().map_or(0, |r| r.len());
    let (mut beta, mut u) = match start { Some((b, c)) => (b, if c > 1e-6 { (c.exp() - 1.0).max(1e-12).ln() } else { -10.0 }), None => (vec![0.0; k], 0.0) };
    let eval = |b: &[f64], u: f64| -> (f64, Vec<f64>) {
        let m: Vec<f64> = x.iter().map(|r| r.iter().zip(b).map(|(a, w)| a * w).sum()).collect();
        let (l, gm, gc) = nll_grad(&m, n, softplus(u));
        let mut g = vec![0.0; k + 1];
        for (j, r) in x.iter().enumerate() { for i in 0..k { g[i] += gm[j] * r[i]; } }
        g[k] = gc * sig(u);
        (l, g)
    };
    let (mut l, mut g) = eval(&beta, u);
    for _ in 0..60 {
        let d = k + 1;
        let mut h = vec![vec![0.0; d]; d];
        let eps = 1e-5;
        for i in 0..d {
            let (mut b_try, mut u_try) = (beta.clone(), u);
            if i < k { b_try[i] += eps; } else { u_try += eps; }
            let (_, g_try) = eval(&b_try, u_try);
            for j in 0..d { h[j][i] = (g_try[j] - g[j]) / eps; }
        }
        for i in 0..d { for j in 0..i { let a = 0.5 * (h[i][j] + h[j][i]); h[i][j] = a; h[j][i] = a; } h[i][i] += 1e-9; }
        let step = solve(&h, &g).unwrap_or_else(|| g.clone());
        let mut t = 1.0;
        let mut moved = false;
        while t > 1e-6 {
            let b_try: Vec<f64> = beta.iter().enumerate().map(|(i, b)| b - t * step[i]).collect();
            let u_try = u - t * step[k];
            let (l_try, g_try) = eval(&b_try, u_try);
            if l_try <= l { let done = (l - l_try).abs() < 1e-10 * (1.0 + l.abs()); beta = b_try; u = u_try; l = l_try; g = g_try; moved = true; if done { return (beta, softplus(u), l); } break; }
            t *= 0.5;
        }
        if !moved { break; }
    }
    (beta, softplus(u), l)
}

fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a.iter().zip(b).map(|(r, x)| { let mut r = r.clone(); r.push(*x); r }).collect();
    for c in 0..n {
        let p = (c..n).max_by(|i, j| m[*i][c].abs().partial_cmp(&m[*j][c].abs()).unwrap())?;
        if m[p][c].abs() < 1e-14 { return None; }
        m.swap(c, p);
        for r in 0..n { if r != c { let f = m[r][c] / m[c][c]; for k in c..=n { m[r][k] -= f * m[c][k]; } } }
    }
    Some((0..n).map(|i| m[i][n] / m[i][i]).collect())
}

fn per_position_loss(c: &Corpus, m: &[f64], s: f64, band: f64) -> Vec<f64> {
    let lg: Vec<[f64; 3]> = m.iter().map(|x| { let p = probs(s * x, band); [-p[0].ln(), -p[1].ln(), -p[2].ln()] }).collect();
    c.pos_group.iter().zip(&c.pos_out).map(|(g, o)| lg[*g as usize][*o as usize]).collect()
}

/// The standard error of the mean of `diff` over the positions, the games as clusters.
fn clustered_se(c: &Corpus, diff: &[f64]) -> f64 {
    let n = diff.len() as f64;
    let mean = diff.iter().sum::<f64>() / n;
    let mut per: HashMap<u32, (f64, f64)> = HashMap::new();
    for (d, g) in diff.iter().zip(&c.pos_game) { let e = per.entry(*g).or_insert((0.0, 0.0)); e.0 += d; e.1 += 1.0; }
    (per.values().map(|(s, k)| (s - k * mean).powi(2)).sum::<f64>()).sqrt() / n
}

fn gcd(a: i64, b: i64) -> i64 { if b == 0 { a.abs() } else { gcd(b, a % b) } }

/// The integer vectors of size-sum `price` with the signs `sign`, nearest the shares of |beta|.
fn nearest(beta: &[f64], sign: &[i64], price: u32) -> Vec<Vec<i64>> {
    let k = beta.len();
    let tot: f64 = beta.iter().map(|b| b.abs()).sum();
    let share: Vec<f64> = beta.iter().map(|b| if tot > 0.0 { b.abs() / tot * price as f64 } else { price as f64 / k as f64 }).collect();
    let lo: Vec<i64> = share.iter().map(|x| x.floor() as i64).collect();
    let mut out = std::collections::BTreeSet::new();
    for up in 0..(1u32 << k) {
        let w: Vec<i64> = (0..k).map(|i| lo[i] + ((up >> i) & 1) as i64).collect();
        if w.iter().sum::<i64>() == price as i64 && w.iter().fold(0, |a, b| gcd(a, *b)) == 1 { out.insert(w.iter().zip(sign).map(|(a, s)| a * s).collect::<Vec<_>>()); }
    }
    out.into_iter().collect()
}

/// Every vector of size-sum `price` with the signs `sign` (no common divisor above one).
fn every(sign: &[i64], price: u32) -> Vec<Vec<i64>> {
    let k = sign.len();
    let mut out = vec![];
    fn rec(k: usize, left: i64, cur: &mut Vec<i64>, out: &mut Vec<Vec<i64>>) {
        if cur.len() == k - 1 { cur.push(left); out.push(cur.clone()); cur.pop(); return; }
        for x in 0..=left { cur.push(x); rec(k, left - x, cur, out); cur.pop(); }
    }
    if k == 0 { return out; }
    rec(k, price as i64, &mut vec![], &mut out);
    out.into_iter().filter(|w| w.iter().fold(0, |a, b| gcd(a, *b)) == 1).map(|w| w.iter().zip(sign).map(|(a, s)| a * s).collect()).collect()
}

/// The miner's answer.
#[derive(Debug, Clone)]
pub struct Fit {
    pub kinds: Vec<String>,
    pub constant: Vec<String>,
    pub beta: Vec<f64>,
    pub band: f64,
    pub logloss_best: f64,
    pub se_best: f64,
    pub worth: Option<Vec<i64>>,
    pub price: u32,
    pub logloss: f64,
    pub excess: f64,
    pub se: f64,
    pub tried: usize,
    pub positions: usize,
    pub games: usize,
    pub groups: usize,
}

/// THE PRICED SEARCH (see the module docstring); `fuse` the price it gives up at.
pub fn search(c: &Corpus, fuse: u32) -> Fit {
    let positions = c.pos_group.len();
    let (beta, band, nll) = fit(&c.vec, &c.n, None);
    let mb: Vec<f64> = c.vec.iter().map(|r| r.iter().zip(&beta).map(|(a, w)| a * w).sum()).collect();
    let refl = per_position_loss(c, &mb, 1.0, band);
    let se_best = clustered_se(c, &refl);
    let sign: Vec<i64> = beta.iter().map(|b| if *b < 0.0 { -1 } else { 1 }).collect();
    let mut out = Fit { kinds: c.kinds.clone(), constant: c.constant.clone(), beta: beta.clone(), band, logloss_best: nll / positions as f64, se_best, worth: None,
                        price: 0, logloss: f64::NAN, excess: f64::NAN, se: f64::NAN, tried: 0, positions, games: c.games, groups: c.n.len() };
    let tot: f64 = beta.iter().map(|b| b.abs()).sum();
    for price in 1..=fuse {
        let start = Some((vec![tot / price as f64], band));       // (the scale where the vector's weights match the best weights' size: no saturated start)
        let mut cands = nearest(&beta, &sign, price);
        if price <= CHECK_PRICE { for w in every(&sign, price) { if !cands.contains(&w) { cands.push(w); } } }
        let mut best: Option<(f64, Vec<i64>, f64, f64)> = None;
        for w in &cands {
            let m: Vec<Vec<f64>> = c.vec.iter().map(|r| vec![r.iter().zip(w).map(|(a, b)| a * *b as f64).sum()]).collect();
            let (s, cc, l) = fit(&m, &c.n, start.clone());
            if best.as_ref().map_or(true, |b| l < b.0) { best = Some((l, w.clone(), s[0], cc)); }
        }
        out.tried += cands.len();
        let Some((l, w, s, cc)) = best else { continue };
        let m: Vec<f64> = c.vec.iter().map(|r| r.iter().zip(&w).map(|(a, b)| a * *b as f64).sum()).collect();
        let diff: Vec<f64> = per_position_loss(c, &m, s, cc).iter().zip(&refl).map(|(a, b)| a - b).collect();
        let se = clustered_se(c, &diff);
        let excess = diff.iter().sum::<f64>() / positions as f64;
        if excess <= se {
            out.worth = Some(w);
            out.price = price;
            out.logloss = l / positions as f64;
            out.excess = excess;
            out.se = se;
            return out;
        }
    }
    out
}
