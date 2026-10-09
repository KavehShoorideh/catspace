//! Maia, through the standalone crate rust/maia-chess (GPL-3.0; packaged 2026-10-09 to share with the community): the
//! loaded networks cached by path.
use shakmaty::Chess;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

static MODELS: OnceLock<Mutex<HashMap<String, Arc<maia_chess::Maia>>>> = OnceLock::new();

fn model(path: &str) -> Result<Arc<maia_chess::Maia>, String> {
    let lock = MODELS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(m) = lock.lock().unwrap().get(path) { return Ok(m.clone()); }
    let m = Arc::new(maia_chess::Maia::load(path)?);
    lock.lock().unwrap().insert(path.to_string(), m.clone());
    Ok(m)
}

/// The policy over the legal moves at temperature `temp`: [(uci, probability)].
pub fn policy(pos: &Chess, model_path: &str, temp: f32) -> Result<Vec<(String, f32)>, String> {
    model(model_path)?.policy(pos, temp)
}
