//! PLAN 79.39 PART F: `Game::turn`, THE SIDE TO MOVE AS THE STATE SAYS IT, WITH NO READING OF THE END (SYSTEM_SPEC 11 "Layer 2 answers whose turn it is without asking whether the game has ended"; piece: rules; ledger row 46).
//! This file names the games on purpose (a test file may). For each of the seven games the kernel plays, along seeded random games, at every state not ended `turn` equals `mover`; at an ended state it gives a role (the one the state's
//! own side to move names, read here from the position's text) and `mover` gives none. A made game of four roles, turns passing in order, shows the default answers the right role at every state not ended, a game that gives only `mover` gets the default.
use cs_game::kernel::KernelGame;
use cs_game::rng::Rng;
use cs_game::Game;

const GAMES: [&str; 7] = ["chess", "kingofthehill", "3check", "chess960", "horde", "atomic", "antichess"];

fn side_in_text(g: &KernelGame, s: &<KernelGame as Game>::State) -> usize { if g.fen(s).split(' ').nth(1) == Some("w") { 0 } else { 1 } }

#[test]
fn the_kernels_turn_equals_mover_where_the_game_goes_on_and_names_the_side_at_an_end() {
    let (mut live, mut ended) = (0u64, 0u64);
    for (gi, name) in GAMES.iter().enumerate() {
        for seed in 0..60u64 {
            let g = KernelGame::new(name, (seed * 37 % 960) as u32).unwrap();
            let mut rng = Rng::new(1000 * gi as u64 + seed);
            let mut s = g.initial();
            for _ in 0..400 {
                let t = g.turn(&s);
                assert!(t < g.roles().len(), "{name}: a role among roles()");
                assert_eq!(t, side_in_text(&g, &s), "{name}: the side the state says");
                match g.mover(&s) {
                    Some(m) => { live += 1; assert_eq!(t, m, "{name}: at a state not ended, turn is mover"); }
                    None => { ended += 1; assert!(g.outcome(&s).is_some()); break; }
                }
                let ms = g.moves(&s);
                s = g.apply(&s, &ms[rng.below(ms.len())]);
            }
        }
    }
    assert!(live > 1000 && ended > 0, "the samples held both live and ended states: {live} live, {ended} ended");
}

/// A made game of four roles: the turn passes in order; the game ends after `limit` moves.
struct Round { limit: usize }
impl Game for Round {
    type State = usize;
    type Move = u8;
    fn name(&self) -> String { "round".into() }
    fn roles(&self) -> Vec<String> { vec!["a".into(), "b".into(), "c".into(), "d".into()] }
    fn initial(&self) -> usize { 0 }
    fn mover(&self, s: &usize) -> Option<usize> { if *s >= self.limit { None } else { Some(*s % 4) } }
    fn moves(&self, s: &usize) -> Vec<u8> { if *s >= self.limit { vec![] } else { vec![0] } }
    fn apply(&self, s: &usize, _m: &u8) -> usize { s + 1 }
    fn outcome(&self, s: &usize) -> Option<Vec<i64>> { if *s >= self.limit { Some(vec![25; 4]) } else { None } }
    fn move_name(&self, _s: &usize, _m: &u8) -> String { "step".into() }
    fn key(&self, s: &usize) -> u64 { *s as u64 }
}

#[test]
fn the_default_turn_of_a_game_of_four_roles_is_the_role_to_move_at_every_state_not_ended() {
    let g = Round { limit: 10 };
    let mut s = g.initial();
    for ply in 0..10 {
        assert_eq!(g.turn(&s), ply % 4, "ply {ply}");
        assert_eq!(Some(g.turn(&s)), g.mover(&s));
        s = g.apply(&s, &0);
    }
    assert!(g.mover(&s).is_none());
    assert!(g.turn(&s) < g.roles().len(), "at an end the default still names a role of the game");
}

#[test]
fn a_game_that_forwards_only_mover_gets_the_default_turn() {
    let k = KernelGame::new("chess", 0).unwrap();
    let s = k.apply(&k.initial(), &k.moves(&k.initial())[0]);
    assert_eq!(k.turn(&s), 1);
    let w = Wrap(k);
    assert_eq!(w.turn(&s), 1);
    // the wrapper forwards `mover` only: its `turn` is the default, built from `mover`
    assert_eq!(w.mover(&s), Some(1));
}

/// A wrapper that forwards only what the test needs: the default `turn` is built from the forwarded `mover`.
struct Wrap(KernelGame);
impl Game for Wrap {
    type State = <KernelGame as Game>::State;
    type Move = <KernelGame as Game>::Move;
    fn name(&self) -> String { self.0.name() }
    fn roles(&self) -> Vec<String> { self.0.roles() }
    fn initial(&self) -> Self::State { self.0.initial() }
    fn mover(&self, s: &Self::State) -> Option<usize> { self.0.mover(s) }
    fn moves(&self, s: &Self::State) -> Vec<Self::Move> { self.0.moves(s) }
    fn apply(&self, s: &Self::State, m: &Self::Move) -> Self::State { self.0.apply(s, m) }
    fn outcome(&self, s: &Self::State) -> Option<Vec<i64>> { self.0.outcome(s) }
    fn move_name(&self, s: &Self::State, m: &Self::Move) -> String { self.0.move_name(s, m) }
    fn key(&self, s: &Self::State) -> u64 { self.0.key(s) }
}
