pub mod assembly;
pub mod docking;

use pyo3::prelude::*;

pub fn register_lattice_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(docking::enumerate_interface_contacts, m)?)?;
    m.add_function(wrap_pyfunction!(assembly::build_lattice, m)?)?;
    Ok(())
}
