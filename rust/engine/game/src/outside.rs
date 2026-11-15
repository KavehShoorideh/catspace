//! THE UCI TEACHER (plan 78.4): an outside engine that speaks UCI -- the chess family's engine protocol, so it lives beside the chess kernel (rules ingestion's
//! side of the stack, not a layer) -- driven over its pipes. The learning layer sees it only as a `Teacher` (cs_learn::watch): a source of moves to watch.
use crate::kernel::KernelGame;
use crate::placed::Placed;
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
// none of its rules).

fn kind_word(r: shakmaty::Role) -> &'static str {
    match r { shakmaty::Role::Pawn => "pawn", shakmaty::Role::Knight => "knight", shakmaty::Role::Bishop => "bishop",
              shakmaty::Role::Rook => "rook", shakmaty::Role::Queen => "queen", shakmaty::Role::King => "king" }
}

fn kind_role(word: &str) -> Option<shakmaty::Role> {
    Some(match word { "pawn" => shakmaty::Role::Pawn, "knight" => shakmaty::Role::Knight, "bishop" => shakmaty::Role::Bishop,
                      "rook" => shakmaty::Role::Rook, "queen" => shakmaty::Role::Queen, "king" => shakmaty::Role::King, _ => return None })
}

fn colour_of(owner: Role) -> Color { if owner == 0 { Color::White } else { Color::Black } }

/// PLAN 78.43 RULES INGESTION OF THE PLACED READING: the chess family's attack and step tables from shakmaty's bitboards, the board as it stands (a mirror of the codec's
/// `cells`, `bearing_from` and `reach_from` that the Python question sheet reads).
impl Placed for KernelGame {
    fn cells(&self, s: &VariantPosition) -> Vec<(Role, String, String)> {
        s.board().iter().map(|(sq, p)| (if p.color == Color::White { 0 } else { 1 }, kind_word(p.role).to_string(), sq.to_string())).collect()
    }
    fn bearing_from(&self, s: &VariantPosition, owner: Role, kind: &str, at: &str) -> Vec<String> {
        use shakmaty::attacks;
        let (Some(role), Ok(sq)) = (kind_role(kind), shakmaty::Square::from_ascii(at.as_bytes())) else { return vec![] };
        let occ = s.board().occupied();
        let bb = match role {
            shakmaty::Role::Pawn => attacks::pawn_attacks(colour_of(owner), sq),
            shakmaty::Role::Knight => attacks::knight_attacks(sq),
            shakmaty::Role::King => attacks::king_attacks(sq),
            shakmaty::Role::Bishop => attacks::bishop_attacks(sq, occ),
            shakmaty::Role::Rook => attacks::rook_attacks(sq, occ),
            shakmaty::Role::Queen => attacks::queen_attacks(sq, occ),
        };
        bb.into_iter().map(|q| q.to_string()).collect()
    }
    fn reach_from(&self, s: &VariantPosition, owner: Role, kind: &str, at: &str) -> Vec<String> {
        let (Some(role), Ok(sq)) = (kind_role(kind), shakmaty::Square::from_ascii(at.as_bytes())) else { return vec![] };
        let colour = colour_of(owner);
        let board = s.board();
        let foes = board.by_color(!colour) & !board.kings();
        let empty = !board.occupied();
        let mut bits = shakmaty::Bitboard::EMPTY;
        for q in self.bearing_from(s, owner, kind, at) { if let Ok(x) = shakmaty::Square::from_ascii(q.as_bytes()) { bits = bits.with(x); } }
        let mut reach = if role == shakmaty::Role::Pawn { bits & foes } else { bits & (empty | foes) };
        if role == shakmaty::Role::Pawn {
            if let Some(ahead) = sq.offset(if colour == Color::White { 8 } else { -8 }) {
                if empty.contains(ahead) { reach = reach.with(ahead); }
            }
        }
        reach.into_iter().map(|q| q.to_string()).collect()
    }
    fn move_ends(&self, _s: &VariantPosition, m: &Move) -> (String, String) {
        let u = m.to_uci(self.mode).to_string();
        (u[..2].to_string(), u[2..4].to_string())
    }
}
