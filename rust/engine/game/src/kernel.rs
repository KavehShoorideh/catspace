//! THE CHESS KERNELS (plan 78.1): shakmaty's positions for the perfect-information chess variants, behind the `Game` trait. RULES INGESTION, the one chess module
//! of the layer stack: what the variants' rules are is shakmaty's (certified against a GDL text where one is written); the terminal rules mirror the spec's
//! (rules/chess.gdl: the hundred-halfmove draw is terminal). Variants by lichess's names: chess, chess960, crazyhouse, atomic, antichess, kingofthehill, 3check,
//! racingkings, horde. Repetition is the game record's, not the state's: a harness that keeps the record reads it.
use crate::{Fact, Game, Role};
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::variant::{Variant, VariantPosition};
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{CastlingMode, CastlingSide, Color, EnPassantMode, Move, Outcome, Position};

/// The variants this kernel plays (lichess's names; chess960 is chess with its castling mode and a start drawn from the 960).
pub const VARIANTS: [&str; 9] = ["chess", "chess960", "crazyhouse", "atomic", "antichess", "kingofthehill", "3check", "racingkings", "horde"];

/// The halfmove clock at which the game is drawn (the spec's terminal `draw_by_fifty_moves`, read as terminal by both).
pub const CLOCK_LIMIT: u32 = 100;

#[derive(Clone)]
pub struct KernelGame {
    pub variant: Variant,
    pub mode: CastlingMode,
    pub start: VariantPosition,
    pub label: String,
}

impl KernelGame {
    /// The game by its name: a variant of `VARIANTS`; "chess960" takes the start numbered `start960` (0..959, Scharnagl's numbering; 518 is the standard start).
    pub fn new(name: &str, start960: u32) -> Result<KernelGame, String> {
        if name == "chess960" {
            let fen = chess960_fen(start960 % 960);
            return KernelGame::from_fen("chess960", &fen);
        }
        let variant = Variant::from_ascii(name.as_bytes()).map_err(|_| format!("no variant {name}"))?;
        Ok(KernelGame { variant, mode: CastlingMode::Standard, start: VariantPosition::new(variant), label: name.to_string() })
    }

    /// The game started from `fen` (a variant of `VARIANTS`; chess960 reads its castling rights by file).
    pub fn from_fen(name: &str, fen: &str) -> Result<KernelGame, String> {
        let (variant, mode) = if name == "chess960" { (Variant::Chess, CastlingMode::Chess960) } else {
            (Variant::from_ascii(name.as_bytes()).map_err(|_| format!("no variant {name}"))?, CastlingMode::Standard)
        };
        let f: Fen = fen.parse().map_err(|e| format!("fen {fen}: {e}"))?;
        let start = VariantPosition::from_setup(variant, f.into_setup(), mode).map_err(|e| format!("position {fen}: {e}"))?;
        Ok(KernelGame { variant, mode, start, label: name.to_string() })
    }

    /// The state's FEN (the harness's and the trace's).
    pub fn fen(&self, s: &VariantPosition) -> String { Fen::from_position(s.clone(), EnPassantMode::Legal).to_string() }

    /// The legal move named `uci` on `s`, or None.
    pub fn parse(&self, s: &VariantPosition, uci: &str) -> Option<Move> {
        UciMove::from_ascii(uci.as_bytes()).ok().and_then(|u| u.to_move(s).ok())
    }
}

impl Game for KernelGame {
    type State = VariantPosition;
    type Move = Move;
    fn name(&self) -> String { self.label.clone() }
    fn roles(&self) -> Vec<String> { vec!["white".into(), "black".into()] }
    fn initial(&self) -> VariantPosition { self.start.clone() }
    fn mover(&self, s: &VariantPosition) -> Option<Role> {
        if self.outcome(s).is_some() { return None; }
        Some(if s.turn() == Color::White { 0 } else { 1 })
    }
    fn moves(&self, s: &VariantPosition) -> Vec<Move> {
        if self.outcome(s).is_some() { return vec![]; }
        s.legal_moves().into_iter().collect()
    }
    fn apply(&self, s: &VariantPosition, m: &Move) -> VariantPosition {
        let mut q = s.clone();
        q.play_unchecked(m);
        q
    }
    fn outcome(&self, s: &VariantPosition) -> Option<Vec<i64>> {
        let o = s.outcome().or_else(|| if s.halfmoves() >= CLOCK_LIMIT { Some(Outcome::Draw) } else { None })?;
        Some(match o {
            Outcome::Decisive { winner: Color::White } => vec![100, 0],
            Outcome::Decisive { winner: Color::Black } => vec![0, 100],
            Outcome::Draw => vec![50, 50],
        })
    }
    fn move_name(&self, _s: &VariantPosition, m: &Move) -> String { m.to_uci(self.mode).to_string() }
    fn key(&self, s: &VariantPosition) -> u64 { s.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0 }
    fn captured(&self, s: &VariantPosition, m: &Move) -> Vec<(Role, String)> {
        if self.variant == Variant::Atomic {                                  // (an explosion takes more than the captured unit: the trait's own reading)
            let me = if s.turn() == Color::White { 0 } else { 1 };
            let before: Vec<(Role, String)> = self.units(s).into_iter().filter(|(o, _)| *o != me).collect();
            let mut left: Vec<(Role, String)> = self.units(&self.apply(s, m)).into_iter().filter(|(o, _)| *o != me).collect();
            return before.into_iter().filter(|u| if let Some(i) = left.iter().position(|x| x == u) { left.swap_remove(i); false } else { true }).collect();
        }
        match m.capture() { Some(r) => vec![(if s.turn() == Color::White { 1 } else { 0 }, r.char().to_string())], None => vec![] }
    }
    fn taken_at(&self, s: &VariantPosition, m: &Move) -> Vec<(Role, String, String)> {
        if self.variant == Variant::Atomic { return self.captured(s, m).into_iter().map(|(o, k)| (o, k, String::new())).collect(); }
        let them = if s.turn() == Color::White { 1 } else { 0 };
        match (m.capture(), m) {
            (Some(r), Move::EnPassant { from, to }) => vec![(them, r.char().to_string(), shakmaty::Square::from_coords(to.file(), from.rank()).to_string())],
            (Some(r), _) => vec![(them, r.char().to_string(), m.to().to_string())],
            (None, _) => vec![],
        }
    }
    fn changed(&self, s: &VariantPosition, m: &Move) -> Vec<(String, String)> {
        if self.variant == Variant::Crazyhouse { return crate::units_changed(self, s, m); }
        match m.promotion() { Some(r) => vec![("p".into(), r.char().to_string())], None => vec![] }
    }
    fn pass(&self, s: &VariantPosition) -> Option<VariantPosition> {
        if s.is_check() || self.outcome(s).is_some() { return None; }
        let mut st = s.clone().into_setup(EnPassantMode::Always);
        st.turn = !st.turn;
        st.ep_square = None;
        VariantPosition::from_setup(self.variant, st, self.mode).ok()
    }
    /// RULES INGESTION OF THE FACT LAYER: the position as the spec's own `true` fluents (rules/chess.gdl: cell, control, castling_right, ep_square, halfmove -- the
    /// clock saturating at 100 -- and, for three-check, the checks each side has given). Only the variants with a written description give facts; the others none.
    /// Certified against the description by co-play (certify::coplay compares these with the oracle's fluents at every state).
    fn base_facts(&self, s: &VariantPosition) -> Vec<Fact> {
        if !matches!(self.variant, Variant::Chess | Variant::KingOfTheHill | Variant::ThreeCheck) { return vec![]; }
        let side = |c: Color| if c == Color::White { "white" } else { "black" };
        let mut out = vec![];
        for (sq, p) in s.board().iter() {
            let kind = match p.role { shakmaty::Role::Pawn => "pawn", shakmaty::Role::Knight => "knight", shakmaty::Role::Bishop => "bishop",
                                      shakmaty::Role::Rook => "rook", shakmaty::Role::Queen => "queen", shakmaty::Role::King => "king" };
            out.push(Fact { relation: "cell".into(), args: vec![sq.file().char().to_string(), sq.rank().char().to_string(), side(p.color).into(), kind.into()] });
        }
        out.push(Fact { relation: "control".into(), args: vec![side(s.turn()).into()] });
        for c in [Color::White, Color::Black] {
            for (cs, wing) in [(CastlingSide::KingSide, "kingside"), (CastlingSide::QueenSide, "queenside")] {
                if s.castles().has(c, cs) { out.push(Fact { relation: "castling_right".into(), args: vec![side(c).into(), wing.into()] }); }
            }
        }
        if let Some(sq) = s.ep_square(EnPassantMode::Always) { out.push(Fact { relation: "ep_square".into(), args: vec![sq.file().char().to_string(), sq.rank().char().to_string()] }); }
        out.push(Fact { relation: "halfmove".into(), args: vec![s.halfmoves().min(CLOCK_LIMIT).to_string()] });
        if self.variant == Variant::ThreeCheck {
            // (the description counts a check one state late -- "a check stands at a state whose side to move is in check, and is counted into the next state" --
            // so while the checked side is to move, the check just given is not yet in the count)
            if let Some(rc) = s.clone().into_setup(EnPassantMode::Always).remaining_checks {
                for (c, r) in [(Color::White, rc.white), (Color::Black, rc.black)] {
                    let left: u32 = r.into();
                    let pending = s.is_check() && s.turn() == !c;
                    out.push(Fact { relation: "checks".into(), args: vec![side(c).into(), (3 - left - pending as u32).to_string()] });
                }
            }
        }
        out.sort();
        out
    }
    fn units(&self, s: &VariantPosition) -> Vec<(Role, String)> {
        s.board().iter().map(|(_sq, p)| (if p.color == Color::White { 0 } else { 1 }, p.role.char().to_string())).collect()
    }
}

/// The Chess960 start numbered `n` (Scharnagl's numbering: the light bishop, the dark bishop, the queen, the knights, then rook, king, rook in the files left).
pub fn chess960_fen(n: u32) -> String {
    const KN: [(usize, usize); 10] = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)];
    let mut rank = ['.'; 8];
    let mut n = n as usize;
    rank[2 * (n % 4) + 1] = 'b';
    n /= 4;
    rank[2 * (n % 4)] = 'b';
    n /= 4;
    let free = |r: &[char; 8]| (0..8).filter(|&i| r[i] == '.').collect::<Vec<_>>();
    let q = free(&rank)[n % 6];
    rank[q] = 'q';
    n /= 6;
    let f = free(&rank);
    let (a, b) = KN[n];
    rank[f[a]] = 'n';
    rank[f[b]] = 'n';
    let f = free(&rank);
    rank[f[0]] = 'r';
    rank[f[1]] = 'k';
    rank[f[2]] = 'r';
    let black: String = rank.iter().collect();
    format!("{black}/pppppppp/8/8/8/8/PPPPPPPP/{} w KQkq - 0 1", black.to_uppercase())
}

/// THE CHESS BINDING OF THE SPEC's MOVE TERMS (rules ingestion, beside the kernel): the kernel's UCI name of a move -> rules/chess.gdl's `does` term
/// ((move f1 r1 f2 r2) / (promote f1 r1 f2 r2 kind)), the names certify compares.
pub fn uci_to_spec_term(uci: &str) -> String {
    let c: Vec<char> = uci.chars().collect();
    if c.len() == 5 {
        let kind = match c[4] { 'q' => "queen", 'r' => "rook", 'b' => "bishop", 'n' => "knight", _ => "?" };
        return format!("(promote {} {} {} {} {kind})", c[0], c[1], c[2], c[3]);
    }
    format!("(move {} {} {} {})", c[0], c[1], c[2], c[3])
}
