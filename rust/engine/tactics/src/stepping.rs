//! PLAN 79.33 (SYSTEM_SPEC 11 "What plan 79.32 left of one algorithm", 13 item 51; piece: the tactical reasoner as cases of the base). A game's positions as the node of layer 0's general walk (`cs_base::choice::fold::Node`): the positions of the line being walked, the game's moves as the options, the side that asks picking at its own turn.
//! INTERFACE. `Stepped::new(game, side, state)`: the walk starts at `state`; `top()` the position the walk is at. Called by: `toward` (the search toward claims) and `lines::settle_tail` (the tail of a line); each hands layer 0 a kind of answer and walks this node.
use cs_base::choice::fold::Node;
use cs_game::{Game, Role};

pub struct Stepped<'a, G: Game> {
    pub g: &'a G,
    pub side: Role,
    pub stack: Vec<G::State>,
}

impl<'a, G: Game> Stepped<'a, G> {
    pub fn new(g: &'a G, side: Role, state: &G::State) -> Self { Stepped { g, side, stack: vec![state.clone()] } }
    #[inline(always)]
    pub fn top(&self) -> &G::State { self.stack.last().expect("a position is current") }
}

impl<G: Game> Node for Stepped<'_, G> {
    type Opt = G::Move;
    #[inline(always)]
    fn own_turn(&self) -> bool { self.g.mover(self.top()) == Some(self.side) }
    #[inline(always)]
    fn options(&mut self, out: &mut Vec<G::Move>) { *out = self.g.moves(self.top()); }
    #[inline(always)]
    fn enter(&mut self, m: &G::Move) { let q = self.g.apply(self.top(), m); self.stack.push(q); }
    #[inline(always)]
    fn leave(&mut self) { self.stack.pop(); }
}
