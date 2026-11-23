//! PLAN 78.43 THE PLACED READING OF THE KERNEL, AND (plan 79.23) ITS TEXT DOOR AND THE DESCRIPTION'S NOTATION: rules ingestion beside the kernel (the outside engine's driver moved to cs_learn::outside).
use crate::kernel::KernelGame;
use crate::placed::{Kind, NoStep, Place, Placed};
use crate::rules::{DescriptionNames, StateText};
use crate::{Game, Role};
use crate::starts::Starts;
use shakmaty::variant::{Variant, VariantPosition};
use shakmaty::{Color, EnPassantMode, Move, Position};

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
        // PLAN 79.35 (what `may_move` means): a leader of the other side may be taken only in a game where a leader is an ordinary unit or is taken by a removal (antichess, atomic); and in atomic a leader takes nothing
        let leader_takable = matches!(self.variant, Variant::Antichess | Variant::Atomic);
        let foes = if leader_takable { board.by_color(!colour) } else { board.by_color(!colour) & !board.kings() };
        let foes = if self.variant == Variant::Atomic && role == shakmaty::Role::King { shakmaty::Bitboard::EMPTY } else { foes };
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
    fn kind_at(&self, s: &VariantPosition, at: Place) -> Option<Kind> { s.board().role_at(square_of(at)).map(kind_of) }
    fn find_step(&self, s: &VariantPosition, from: Place, to: Place) -> Option<Move> {
        // the rules' own legal moves, a board the game calls over included (the Python codec asks the board, not the game's outcome; callers that need a live position test the outcome first)
        let mut found: Vec<Move> = s.legal_moves().into_iter().filter(|m| self.move_ends(s, m) == (from, to)).collect();
        found.sort_by_key(|m| m.promotion() != Some(shakmaty::Role::Queen));
        found.into_iter().next()
    }
    fn why_not(&self, s: &VariantPosition, from: Place, to: Place) -> (NoStep, Vec<Place>) {
        let (f, t) = (square_of(from), square_of(to));
        let board = s.board();
        let (p, occ) = (board.piece_at(f), board.piece_at(t));
        let Some(p) = p.filter(|p| p.color == s.turn()) else { return (NoStep::UnitGone, vec![]) };
        if occ.is_some_and(|o| o.color == p.color) { return (NoStep::PlaceTaken, vec![to]); }
        if p.role == shakmaty::Role::Pawn && occ.is_some() && f.file() == t.file() { return (NoStep::PlaceTaken, vec![to]); }
        if matches!(p.role, shakmaty::Role::Bishop | shakmaty::Role::Rook | shakmaty::Role::Queen) {
            let between = shakmaty::attacks::between(f, t) & board.occupied();
            if between.any() { return (NoStep::LineShut, between.into_iter().map(place_of).collect()); }
        }
        (NoStep::MayNotMove, vec![from])
    }
    fn move_info(&self, _s: &VariantPosition, m: &Move) -> (Kind, Option<Kind>) { (kind_of(m.role()), m.capture().map(kind_of)) }
    fn strikes_place(&self, s: &VariantPosition, owner: Role, at: Place) -> bool {
        let board = s.board();
        board.attacks_to(square_of(at), colour_of(owner), board.occupied()).any()
    }
    fn without(&self, s: &VariantPosition, at: Place) -> Option<VariantPosition> {
        self.edited(s, |b| { b.discard_piece_at(square_of(at)); Some(()) })
    }
    fn relocate(&self, s: &VariantPosition, from: Place, to: Place) -> Option<VariantPosition> {
        self.edited(s, |b| { let p = b.remove_piece_at(square_of(from))?; b.set_piece_at(square_of(to), p); Some(()) })
    }
}

/// PLAN 79.35 PART B: the game with a start by number is the one named "chess960" (960 starts, Scharnagl's numbering, kernel.rs); every other game has one start.
impl Starts for KernelGame {
    fn start_count(&self) -> usize { if self.label == "chess960" { 960 } else { 1 } }
    fn with_start(&self, n: usize) -> Result<KernelGame, String> {
        if n >= self.start_count() { return Err(format!("{} has {} start(s), none numbered {n}", self.label, self.start_count())); }
        if self.start_count() == 1 { Ok(self.clone()) } else { KernelGame::new(&self.label, n as u32) }
    }
}

/// PLAN 79.23: the kernel's move name in the notation of the description (the binding kernel.rs keeps for the co-play).
impl DescriptionNames for KernelGame {
    fn description_name(&self, name: &str) -> String { crate::kernel::uci_to_spec_term(name) }
}

/// PLAN 79.23 THE TEXT DOOR OF THE KERNEL (rules ingestion): the state as the game's text form, keeping the square a unit passed over on a two-step advance after every such advance (the description's state holds it; the kernel's own writer drops it where no taking is legal), and read back.
impl StateText for KernelGame {
    fn state_text(&self, s: &VariantPosition) -> String { shakmaty::fen::Fen::from_setup(s.clone().into_setup(EnPassantMode::Always)).to_string() }
    fn read_state_text(&self, text: &str) -> Result<VariantPosition, String> {
        let f: shakmaty::fen::Fen = text.parse().map_err(|e| format!("text {text}: {e}"))?;
        VariantPosition::from_setup(self.variant, f.into_setup(), self.mode).map_err(|e| format!("position {text}: {e}"))
    }
}

impl KernelGame {
    /// A hypothetical board: `s` with its pieces edited, the en passant right gone, the castling rights kept only where the king and the rook still stand, and the result
    /// accepted only if the kernel takes it as a position.
    fn edited(&self, s: &VariantPosition, f: impl FnOnce(&mut shakmaty::Board) -> Option<()>) -> Option<VariantPosition> {
        let mut setup = s.clone().into_setup(EnPassantMode::Always);
        f(&mut setup.board)?;
        setup.ep_square = None;
        setup.castling_rights = match shakmaty::Castles::from_setup(&setup, self.mode) { Ok(c) | Err(c) => c.castling_rights() };
        VariantPosition::from_setup(self.variant, setup, self.mode).ok()
    }
}
