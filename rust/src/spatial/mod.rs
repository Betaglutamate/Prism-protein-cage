pub mod kdtree;
pub mod sasa;

use pyo3::prelude::*;

pub fn register_spatial_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kdtree::clash_check, m)?)?;
    m.add_function(wrap_pyfunction!(kdtree::find_contacts, m)?)?;
    m.add_function(wrap_pyfunction!(sasa::compute_sasa, m)?)?;
    Ok(())
}
