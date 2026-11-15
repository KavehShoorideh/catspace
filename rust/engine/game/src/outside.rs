//! THE UCI TEACHER (plan 78.4): an outside engine that speaks UCI -- the chess family's engine protocol, so it lives beside the chess kernel (rules ingestion's
//! side of the stack, not a layer) -- driven over its pipes. The learning layer sees it only as a `Teacher` (cs_learn::watch): a source of moves to watch.
use crate::kernel::KernelGame;
use crate::placed::{Kind, Place, Placed};
use crate::Role;
use shakmaty::variant::VariantPosition;
use shakmaty::{Color, Move, Position};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

pub struct Uci { child: Child, input: ChildStdin, output: BufReader<ChildStdout> }

impl Uci {
    pub fn start(path: &str, options: &[(String, String)]) -> std::io::Result<Uci> {
        let mut child = Command::new(path).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()?;
        let input = child.stdin.take().unwrap();
        let output = BufReader::new(child.stdout.take().unwrap());
        let mut u = Uci { child, input, output };
        u.send("uci")?;
        u.until("uciok")?;
        for (k, v) in options { u.send(&format!("setoption name {k} value {v}"))?; }
        u.send("isready")?;
        u.until("readyok")?;
        Ok(u)
    }
    pub fn send(&mut self, s: &str) -> std::io::Result<()> { writeln!(self.input, "{s}")?; self.input.flush() }
    fn until(&mut self, word: &str) -> std::io::Result<String> {
        let mut line = String::new();
        loop {
            line.clear();
            if self.output.read_line(&mut line)? == 0 { return Err(std::io::Error::other("the engine closed its output")); }
            if line.starts_with(word) { return Ok(line.trim().to_string()); }
        }
    }
    /// The engine's move after `position <start> moves ...` within `movetime_ms`.
    pub fn best(&mut self, start: &str, moves: &[String], movetime_ms: u64) -> std::io::Result<String> {
        let ms = if moves.is_empty() { String::new() } else { format!(" moves {}", moves.join(" ")) };
        self.send(&format!("position {start}{ms}"))?;
        self.send(&format!("go movetime {movetime_ms}"))?;
        let l = self.until("bestmove")?;
        Ok(l.split_whitespace().nth(1).unwrap_or("0000").to_string())
    }
}

impl Drop for Uci {
    fn drop(&mut self) { let _ = self.send("quit"); let _ = self.child.wait(); }
}


/// TRAINING ONLY (see cs_learn::watch::Teacher): nothing of the decision path may depend on this module.
/// The UCI engine as a game's move source at a fixed move time (the learning layer's `Teacher`, implemented in cs_learn's terms by the caller).
pub struct UciTeacher { pub eng: Uci, pub movetime_ms: u64, pub start: String }

impl UciTeacher {
    pub fn new(path: &str, options: &[(String, String)], movetime_ms: u64) -> Option<UciTeacher> {
        Uci::start(path, options).ok().map(|eng| UciTeacher { eng, movetime_ms, start: "startpos".into() })
    }
    pub fn begin(&mut self, start: &str) { self.start = start.to_string(); let _ = self.eng.send("ucinewgame"); }
    pub fn answer(&mut self, moves: &[String]) -> Option<String> { self.eng.best(&self.start, moves, self.movetime_ms).ok() }
}

// ------------------------------------------------------------------------------------------------------------------------------ the placed reading (plan 78.43)
// Rules ingestion for the question sheet's light reading, kept here beside the kernel (not in kernel.rs: the certificates of the kernel hash that file and this binding changes
// none of its rules). Places and kinds are numbers: the places by the text of their names (file by file), the kinds by the text of their words.

const KIND_NAMES: [&str; 6] = ["bishop", "king", "knight", "pawn", "queen", "rook"];

fn kind_of(r: shakmaty::Role) -> Kind {
    match r { shakmaty::Role::Bishop => 0, shakmaty::Role::King => 1, shakmaty::Role::Knight => 2, shakmaty::Role::Pawn => 3, shakmaty::Role::Queen => 4, shakmaty::Role::Rook => 5 }
}

fn role_of(k: Kind) -> shakmaty::Role {
    match k { 0 => shakmaty::Role::Bishop, 1 => shakmaty::Role::King, 2 => shakmaty::Role::Knight, 3 => shakmaty::Role::Pawn, 4 => shakmaty::Role::Queen, _ => shakmaty::Role::Rook }
}

fn place_of(sq: shakmaty::Square) -> Place { sq.file() as u16 * 8 + sq.rank() as u16 }

fn square_of(p: Place) -> shakmaty::Square { shakmaty::Square::from_coords(shakmaty::File::new(p as u32 / 8), shakmaty::Rank::new(p as u32 % 8)) }

fn colour_of(owner: Role) -> Color { if owner == 0 { Color::White } else { Color::Black } }

fn places_of(bb: shakmaty::Bitboard) -> Vec<Place> {
    let mut v: Vec<Place> = bb.into_iter().map(place_of).collect();
    v.sort_unstable();
    v
}

/// PLAN 78.43 RULES INGESTION OF THE PLACED READING: the chess family's attack and step tables from shakmaty's bitboards, the board as it stands (a mirror of the codec's
/// `cells`, `bearing_from` and `reach_from` that the Python question sheet reads).
impl Placed for KernelGame {
    fn place_names(&self) -> Vec<String> {
        let mut v = vec![];
        for f in 0..8u32 { for r in 0..8u32 { v.push(shakmaty::Square::from_coords(shakmaty::File::new(f), shakmaty::Rank::new(r)).to_string()); } }
        v
    }
    fn kind_names(&self) -> Vec<String> { KIND_NAMES.iter().map(|s| s.to_string()).collect() }
    fn cells(&self, s: &VariantPosition) -> Vec<(Role, Kind, Place)> {
        s.board().iter().map(|(sq, p)| (if p.color == Color::White { 0 } else { 1 }, kind_of(p.role), place_of(sq))).collect()
    }
    fn bearing_from(&self, s: &VariantPosition, owner: Role, kind: Kind, at: Place) -> Vec<Place> {
        use shakmaty::attacks;
        let (role, sq) = (role_of(kind), square_of(at));
        let occ = s.board().occupied();
        let bb = match role {
            shakmaty::Role::Pawn => attacks::pawn_attacks(colour_of(owner), sq),
            shakmaty::Role::Knight => attacks::knight_attacks(sq),
            shakmaty::Role::King => attacks::king_attacks(sq),
            shakmaty::Role::Bishop => attacks::bishop_attacks(sq, occ),
            shakmaty::Role::Rook => attacks::rook_attacks(sq, occ),
            shakmaty::Role::Queen => attacks::queen_attacks(sq, occ),
        };
        places_of(bb)
    }
    fn reach_from(&self, s: &VariantPosition, owner: Role, kind: Kind, at: Place) -> Vec<Place> {
        use shakmaty::attacks;
        let (role, sq) = (role_of(kind), square_of(at));
        let colour = colour_of(owner);
        let board = s.board();
        let foes = board.by_color(!colour) & !board.kings();
        let empty = !board.occupied();
        let occ = board.occupied();
        let bits = match role {
            shakmaty::Role::Pawn => attacks::pawn_attacks(colour, sq),
            shakmaty::Role::Knight => attacks::knight_attacks(sq),
            shakmaty::Role::King => attacks::king_attacks(sq),
            shakmaty::Role::Bishop => attacks::bishop_attacks(sq, occ),
            shakmaty::Role::Rook => attacks::rook_attacks(sq, occ),
            shakmaty::Role::Queen => attacks::queen_attacks(sq, occ),
        };
        let mut reach = if role == shakmaty::Role::Pawn { bits & foes } else { bits & (empty | foes) };
        if role == shakmaty::Role::Pawn {
            if let Some(ahead) = sq.offset(if colour == Color::White { 8 } else { -8 }) {
                if empty.contains(ahead) { reach = reach.with(ahead); }
            }
        }
        places_of(reach)
    }
    fn move_ends(&self, _s: &VariantPosition, m: &Move) -> (Place, Place) {
        let u = m.to_uci(self.mode).to_string();
        match (u.get(..2).and_then(|x| x.parse().ok()), u.get(2..4).and_then(|x| x.parse().ok())) {
            (Some(a), Some(b)) => (place_of(a), place_of(b)),
            _ => (place_of(m.to()), place_of(m.to())),
        }
    }
}
