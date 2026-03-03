use pyo3::prelude::*;

mod geometry;
mod lattice;
mod spatial;

/// PRISM Rust core — high-performance geometry kernels for protein cage design.
///
/// This module is imported in Python as `prism._rust_core`.
#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Geometry submodule
    let geometry_mod = PyModule::new_bound(m.py(), "geometry")?;
    geometry::register_geometry_module(&geometry_mod)?;
    m.add_submodule(&geometry_mod)?;

    // Spatial submodule
    let spatial_mod = PyModule::new_bound(m.py(), "spatial")?;
    spatial::register_spatial_module(&spatial_mod)?;
    m.add_submodule(&spatial_mod)?;

    // Lattice submodule
    let lattice_mod = PyModule::new_bound(m.py(), "lattice")?;
    lattice::register_lattice_module(&lattice_mod)?;
    m.add_submodule(&lattice_mod)?;

    // Top-level convenience re-exports
    m.add_function(wrap_pyfunction!(geometry::symmetry::apply_symmetry_ops, m)?)?;
    m.add_function(wrap_pyfunction!(geometry::cavity::compute_cavity_volume, m)?)?;
    m.add_function(wrap_pyfunction!(geometry::cavity::find_interior_residues, m)?)?;
    m.add_function(wrap_pyfunction!(spatial::kdtree::clash_check, m)?)?;
    m.add_function(wrap_pyfunction!(spatial::sasa::compute_sasa, m)?)?;

    Ok(())
}
