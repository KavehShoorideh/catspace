//! THE MATCHER moved down to layer 0 (plan 79.30; cs_base::logic::matcher, SYSTEM_SPEC 11 "Layer 0 holds six things" item 1). This path hands it on, so no caller changes; the code and its tests are cs_base's.
pub use cs_base::logic::matcher::{bindings, bindings_fused, Binding, FactIndex};
