//! PLAN 79.08 C, THE DECISION LOG (SYSTEM_SPEC 7 "Calculation and estimate, decided together" points 3, 5 and 6: the wake hands the last layer, for each candidate move, what was calculated and the several estimates, and logs those inputs a decision; its inputs stay readable; piece: the wake).
//! ONE RECORD A DECISION, returned by `wake::decide` and written by the caller (this crate writes nothing to disk): for every candidate move at the root what was calculated (the search's value as the null window reads it, held equal or not, how its line ended, whether its own line was read and how it stood), the goal it serves
//! (it is a fix of) or answers (it answers a threat of theirs), and each estimate (the chance of the goal it serves from the table, the reasons' score of the move); with the goals, the threat rows, the named moves and the reason for the choice, all in WORDS (units as "owner kind place" in the game's own words, which the
//! game gave; nothing of any game is named here). This is the learned chooser's training data later; until then the fixed rule of rule.rs decides. Game-free.
//!
//! INTERFACE. `DecisionLog` (below) and `DecisionLog::to_json()`; built by `wake::decide`; read by the binding (`cs_engine`'s wake binding turns it into JSON) and by the tests.
use serde_json::{json, Value};

/// One candidate move at the root.
#[derive(Debug, Clone, Default)]
pub struct MoveLog {
    pub mv: String,
    /// the search's answer at the finished depth to "at least the best?" for this move (a null window), None where the clock stopped before it was asked
    pub value: Option<i64>,
    pub equal: bool,
    /// the table's best line from this move and how it ended (the search's own reading)
    pub line: Vec<String>, pub ended: String,
    /// named to the search (read first and in full) and, if its own line was read, how it stood: the verdict word and the net
    pub named: bool, pub own_read: Option<(String, i64, String)>,
    /// the goals this move is a fix of ("fix of <target> (<fact>)") and the threats of theirs it answers
    pub serves: Vec<String>, pub answers: Vec<String>,
    /// the chance of the best goal it serves, from the table (None: no table, or it serves none) and the reasons' score and how many reasons it carries (None: no reasons given)
    pub chance: Option<f64>, pub reasons_score: Option<f64>, pub reasons_carried: Option<usize>,
    /// PLAN 79.10 A: what their goals say after this move (None: not read; the clock did not reach it)
    pub their_after: Option<TheirAfterLog>,
}

/// PLAN 79.10 A: the other side's goals on the position after one move of ours, in words: the target at the top of their chain of the highest stake (None: no chain of theirs stands after the move), the things in the way of its nearest goal before (None: no such chain stood before) and after, the units the move took out of its way
/// and put in its way, the fixes of theirs named at that position (read first and in full), and the fact in one sentence. A fact about the move: it adds to no value.
#[derive(Debug, Clone, Default)]
pub struct TheirAfterLog { pub top: Option<String>, pub before: Option<usize>, pub after: usize, pub took_out: Vec<String>, pub put_in: Vec<String>, pub names: Vec<String>, pub said: String }

/// A goal in words, with what stops it, its fixes and its estimate.
#[derive(Debug, Clone, Default)]
pub struct GoalLog {
    pub side: String, pub target: String, pub source: String, pub depth: u8, pub parent: Option<String>, pub via: Option<String>, pub unmet: Vec<String>, pub stoppers: Vec<String>, pub fixes: Vec<(String, Vec<String>)>,
    pub in_the_way: usize, pub chance: Option<(f64, u32, String)>, pub chosen: bool, pub goal_unit: bool,
}

/// A row of the threat list in words.
#[derive(Debug, Clone, Default)]
pub struct RowLog { pub list: String, pub unit: String, pub class: String, pub verdict: String, pub net: Option<i64>, pub ended: String, pub line: Vec<String>, pub responses: Vec<(String, String, i64)>, pub carried: bool }

/// The one record of a decision.
#[derive(Debug, Clone, Default)]
pub struct DecisionLog {
    pub moves: Vec<MoveLog>, pub chosen: Option<String>, pub why: String,
    pub chosen_goal: Option<GoalLog>, pub far_goal: Option<String>, pub their_threat: Option<String>,
    pub goals: Vec<GoalLog>, pub rows: Vec<RowLog>,
    pub named_mine: Vec<(String, String, i64, String)>, pub named_theirs: Vec<(String, String, i64, String)>, pub refuted: Vec<(String, bool, String)>, pub unlisted: Vec<(String, bool)>,
    /// what is missing and said so: a piece asked for with nothing to read (no table, no reasons, no learned seeds)
    pub notes: Vec<String>,
    /// PLAN 79.10 A: the moves of ours whose child position the clock did not reach (their goals after them were not read: a gap)
    pub after_not_asked: Vec<String>,
    /// PLAN 79.12 A: the equal reading kept the search's first move (None: the reading did not run: the game was decided, the clock cut it, or no move was first) and, where it did not, what its null-window reading was and by how much the best was above it (None: unknown)
    pub first_held: Option<bool>, pub first_lowered_by: Option<i64>,
    /// PLAN 79.12 A: the wall of each step of the decision (step, what it was, seconds), in the order they ran
    pub step_seconds: Vec<(u8, String, f64)>,
    /// PLAN 79.12 B: the last layer's rows (None: no chooser given)
    pub chooser: Option<ChooserLog>,
}

/// PLAN 79.12 B: what the last layer saw and did in one decision: the mode, the seed, the names of the inputs, the candidates (the moves held equal that the layer may pick among) with their rows and weighted sums, the move the rule would pick, the move played, the propensity of the move played.
#[derive(Debug, Clone, Default)]
pub struct ChooserLog { pub mode: String, pub seed: u64, pub inputs: Vec<String>, pub candidates: Vec<(String, Vec<f64>, f64)>, pub fixed_pick: Option<String>, pub played: Option<String>, pub propensity: f64, pub applied: bool }

fn kept(v: &[(String, String, i64, String)]) -> Value { json!(v.iter().map(|k| json!({"move": k.0, "verdict": k.1, "net": k.2, "ended": k.3})).collect::<Vec<_>>()) }

fn goal_json(x: &GoalLog) -> Value {
    json!({"side": x.side, "target": x.target, "source": x.source, "depth": x.depth, "parent": x.parent, "via": x.via, "unmet": x.unmet, "stoppers": x.stoppers,
           "fixes": x.fixes.iter().map(|f| json!({"kind": f.0, "moves": f.1})).collect::<Vec<_>>(), "in_the_way": x.in_the_way, "chance": x.chance.as_ref().map(|c| json!({"chance": c.0, "n": c.1, "source": c.2})), "chosen": x.chosen, "goal_unit": x.goal_unit})
}

impl DecisionLog {
    /// The record as JSON (words only): the caller writes it where it wants.
    pub fn to_json(&self) -> Value {
        json!({
            "chosen": self.chosen, "why": self.why, "chosen_goal": self.chosen_goal.as_ref().map(goal_json), "far_goal": self.far_goal, "their_threat": self.their_threat,
            "moves": self.moves.iter().map(|m| json!({"move": m.mv, "value": m.value, "equal": m.equal, "line": m.line, "ended": m.ended, "named": m.named,
                "own_read": m.own_read.as_ref().map(|r| json!({"verdict": r.0, "net": r.1, "ended": r.2})), "serves": m.serves, "answers": m.answers, "chance": m.chance, "reasons_score": m.reasons_score, "reasons_carried": m.reasons_carried,
                "their_after": m.their_after.as_ref().map(|a| json!({"top": a.top, "before": a.before, "after": a.after, "took_out": a.took_out, "put_in": a.put_in, "names": a.names, "said": a.said}))})).collect::<Vec<_>>(),
            "goals": self.goals.iter().map(goal_json).collect::<Vec<_>>(),
            "rows": self.rows.iter().map(|r| json!({"list": r.list, "unit": r.unit, "class": r.class, "verdict": r.verdict, "net": r.net, "ended": r.ended, "line": r.line, "carried": r.carried,
                "responses": r.responses.iter().map(|x| json!({"move": x.0, "state": x.1, "net": x.2})).collect::<Vec<_>>()})).collect::<Vec<_>>(),
            "named_mine": kept(&self.named_mine), "named_theirs": kept(&self.named_theirs),
            "refuted": self.refuted.iter().map(|r| json!({"move": r.0, "theirs": r.1, "ended": r.2})).collect::<Vec<_>>(),
            "unlisted": self.unlisted.iter().map(|r| json!({"move": r.0, "theirs": r.1})).collect::<Vec<_>>(),
            "notes": self.notes, "after_not_asked": self.after_not_asked,
            "first_held": self.first_held, "first_lowered_by": self.first_lowered_by,
            "step_seconds": self.step_seconds.iter().map(|x| json!({"step": x.0, "piece": x.1, "seconds": x.2})).collect::<Vec<_>>(),
            "chooser": self.chooser.as_ref().map(|c| json!({"mode": c.mode, "seed": c.seed, "inputs": c.inputs, "candidates": c.candidates.iter().map(|x| json!({"move": x.0, "row": x.1, "sum": x.2})).collect::<Vec<_>>(),
                "fixed_pick": c.fixed_pick, "played": c.played, "propensity": c.propensity, "applied": c.applied})),
        })
    }
}
