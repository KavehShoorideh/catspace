//! PLAN 79.23 PART C: THE UCI TEACHER, moved up out of the rules' crate (it is no part of the rules): an outside engine that speaks the chess family's engine protocol, driven over its pipes. TRAINING ONLY (see `watch::Teacher`):
//! nothing of the decision path may depend on it. Names the protocol's words, so it is one of the files the generic scan of tests/test_plan78.py allows.
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
