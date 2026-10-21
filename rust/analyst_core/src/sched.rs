//! PLAN 56 STAGE 2a: THE SCHEDULER CORE. The task GRAPH (id, kind, parent, waits-on, state, result slot), the ORDER (a binary heap of the ready tasks keyed
//! (queue, priority tuple, the order the task became ready), the lowest key first) and the propagation rules, with exactly the semantics of the Python reference
//! (`discovery/scheduler.py`: `Scheduler.push/submit/_finish`, `route_value`, `best_route`, `question_killed`). The task BODIES stay Python (stage 2b/2c port them);
//! the result slot holds whatever the body returned (a Python object). A task's priority tuple is a list of numbers (f64: the recall score of an agenda question is a
//! rounded float, the rest are small integers, which f64 holds exactly).
//!
//! Life of a task: `push` makes it READY (in the heap); `pop` hands out the best ready one (RUNNING); the running task either `wait`s on children it pushed (WAITING) or is
//! `complete`d (DONE: its result is stored and the parent, if this was the last child it waited on, becomes READY again, queued with its own priority and a fresh order).
//! `reprioritise` moves a READY task (lazy deletion: the old heap entry is skipped when popped); on a task in another state it only records the new priority, used when it next becomes ready.
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State { Ready, Running, Waiting, Done }

struct Rec {
    kind: String,
    parent: Option<usize>,
    waits: usize,                    // children still outstanding
    state: State,
    queue: u8,
    prio: Vec<f64>,
    order: u64,                      // the order it last became ready
    version: u64,                    // bumps on every (re)queue; a heap entry with another version is stale
    depth: u32,
    result: Option<PyObject>,
}

#[derive(PartialEq)]
struct Key { queue: u8, prio: Vec<f64>, order: u64, id: usize, version: u64 }
impl Eq for Key {}
impl Ord for Key {
    fn cmp(&self, o: &Self) -> Ordering {
        self.queue.cmp(&o.queue)
            .then_with(|| {                          // lexicographic like a Python tuple: a prefix is smaller
                for (a, b) in self.prio.iter().zip(o.prio.iter()) {
                    match a.partial_cmp(b).unwrap_or(Ordering::Equal) { Ordering::Equal => {}, x => return x }
                }
                self.prio.len().cmp(&o.prio.len())
            })
            .then(self.order.cmp(&o.order))
            .then(self.id.cmp(&o.id))
    }
}
impl PartialOrd for Key { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }

#[pyclass(name = "Scheduler")]
pub struct Sched {
    tasks: Vec<Rec>,
    heap: BinaryHeap<Reverse<Key>>,
    order: u64,
    ready: usize,
    ready_max: usize,
    steps: u64,
}

impl Sched {
    fn rec(&self, id: usize) -> PyResult<&Rec> {
        self.tasks.get(id).ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("no task {id}")))
    }
    fn enqueue(&mut self, id: usize) {
        self.order += 1;
        let t = &mut self.tasks[id];
        t.state = State::Ready;
        t.order = self.order;
        t.version += 1;
        let k = Key { queue: t.queue, prio: t.prio.clone(), order: t.order, id, version: t.version };
        self.heap.push(Reverse(k));
        self.ready += 1;
        self.ready_max = self.ready_max.max(self.ready);
    }
}

#[pymethods]
impl Sched {
    #[new]
    fn new() -> Self { Sched { tasks: Vec::new(), heap: BinaryHeap::new(), order: 0, ready: 0, ready_max: 0, steps: 0 } }

    /// A new READY task under `parent` (not yet waited on: `wait` does that). -> its id.
    #[pyo3(signature = (kind, parent=None, queue=0, prio=Vec::new()))]
    fn push(&mut self, kind: &str, parent: Option<usize>, queue: u8, prio: Vec<f64>) -> PyResult<usize> {
        let depth = match parent { Some(p) => self.rec(p)?.depth + 1, None => 0 };
        let id = self.tasks.len();
        self.tasks.push(Rec { kind: kind.to_string(), parent, waits: 0, state: State::Ready, queue, prio, order: 0, version: 0, depth, result: None });
        self.enqueue(id);
        Ok(id)
    }

    /// The ready task with the lowest key (queue, priority, order) -> its id, now RUNNING; None when nothing is ready.
    fn pop(&mut self) -> Option<usize> {
        while let Some(Reverse(k)) = self.heap.pop() {
            let t = &mut self.tasks[k.id];
            if t.state != State::Ready || t.version != k.version { continue; }       // stale (reprioritised)
            t.state = State::Running;
            self.ready -= 1;
            self.steps += 1;
            return Some(k.id);
        }
        None
    }

    /// `task` (running) suspends on `child`.
    fn wait(&mut self, task: usize, child: usize) -> PyResult<()> {
        self.rec(child)?;
        let t = self.rec(task)?;
        if t.state != State::Running && t.state != State::Waiting {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("task {task} cannot wait: it is {:?}", t.state)));
        }
        let t = &mut self.tasks[task];
        t.waits += 1;
        t.state = State::Waiting;
        Ok(())
    }

    /// `task` ends with `result` (None for an error: the Python side carries it). Its parent, if this was the last child awaited, becomes READY. -> the parent's id when it was woken.
    #[pyo3(signature = (task, result=None))]
    fn complete(&mut self, task: usize, result: Option<PyObject>) -> PyResult<Option<usize>> {
        self.rec(task)?;
        if self.tasks[task].state == State::Done {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("task {task} is already done")));
        }
        if self.tasks[task].state == State::Ready { self.ready -= 1; }
        let t = &mut self.tasks[task];
        t.state = State::Done;
        t.result = result;
        let Some(p) = t.parent else { return Ok(None) };
        let pr = &mut self.tasks[p];
        if pr.waits > 0 { pr.waits -= 1; }
        if pr.waits == 0 && pr.state == State::Waiting {
            self.enqueue(p);
            return Ok(Some(p));
        }
        Ok(None)
    }

    /// The result slot of a done task (None when empty).
    fn result(&self, py: Python<'_>, task: usize) -> PyResult<Option<PyObject>> {
        Ok(self.rec(task)?.result.as_ref().map(|r| r.clone_ref(py)))
    }

    /// New priority for `task`. A READY task moves in the heap; otherwise the priority is kept for when it next becomes ready.
    #[pyo3(signature = (task, queue, prio))]
    fn reprioritise(&mut self, task: usize, queue: u8, prio: Vec<f64>) -> PyResult<()> {
        self.rec(task)?;
        let t = &mut self.tasks[task];
        t.queue = queue;
        t.prio = prio;
        if t.state == State::Ready {
            let (order, version) = (t.order, t.version + 1);
            t.version = version;
            let k = Key { queue: t.queue, prio: t.prio.clone(), order, id: task, version };
            self.heap.push(Reverse(k));
        }
        Ok(())
    }

    fn state(&self, task: usize) -> PyResult<&'static str> {
        Ok(match self.rec(task)?.state { State::Ready => "ready", State::Running => "running", State::Waiting => "waiting", State::Done => "done" })
    }
    fn kind(&self, task: usize) -> PyResult<String> { Ok(self.rec(task)?.kind.clone()) }
    fn parent(&self, task: usize) -> PyResult<Option<usize>> { Ok(self.rec(task)?.parent) }
    fn depth(&self, task: usize) -> PyResult<u32> { Ok(self.rec(task)?.depth) }
    fn waits(&self, task: usize) -> PyResult<usize> { Ok(self.rec(task)?.waits) }
    fn __len__(&self) -> usize { self.ready }

    /// {tasks, by_kind, ready, ready_max, waiting, running, done, steps}
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let kinds = PyDict::new(py);
        let (mut w, mut r, mut dn) = (0usize, 0usize, 0usize);
        for t in &self.tasks {
            let n: usize = kinds.get_item(&t.kind)?.map(|x| x.extract()).transpose()?.unwrap_or(0);
            kinds.set_item(&t.kind, n + 1)?;
            match t.state { State::Waiting => w += 1, State::Running => r += 1, State::Done => dn += 1, State::Ready => {} }
        }
        d.set_item("tasks", self.tasks.len())?;
        d.set_item("by_kind", kinds)?;
        d.set_item("ready", self.ready)?;
        d.set_item("ready_max", self.ready_max)?;
        d.set_item("waiting", w)?;
        d.set_item("running", r)?;
        d.set_item("done", dn)?;
        d.set_item("steps", self.steps)?;
        Ok(d)
    }
}

/// A route's value: the least over its replies; none read -> 0.
#[pyfunction]
pub fn sched_route_value(gains: Vec<i64>) -> i64 { gains.into_iter().min().unwrap_or(0) }

/// The index of a question's best route: the greatest value, the smallest uci for the tie (the first of equals).
#[pyfunction]
pub fn sched_best_route(values: Vec<i64>, ucis: Vec<String>) -> PyResult<usize> {
    if values.is_empty() || values.len() != ucis.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("best_route: needs equal, non-empty values and ucis"));
    }
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] || (values[i] == values[best] && ucis[i] < ucis[best]) { best = i; }
    }
    Ok(best)
}

/// A chased question is killed when no route claims.
#[pyfunction]
pub fn sched_question_killed(n_claiming: usize) -> bool { n_claiming == 0 }
