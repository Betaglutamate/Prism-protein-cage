pub mod cavity;
pub mod polyhedra;
pub mod symmetry;
pub mod transforms;

use pyo3::prelude::*;

pub fn register_geometry_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(symmetry::get_symmetry_operations, m)?)?;
    m.add_function(wrap_pyfunction!(symmetry::apply_symmetry_ops, m)?)?;
    m.add_function(wrap_pyfunction!(symmetry::get_group_generators, m)?)?;
    m.add_function(wrap_pyfunction!(polyhedra::get_polyhedron_vertices, m)?)?;
    m.add_function(wrap_pyfunction!(polyhedra::get_polyhedron_faces, m)?)?;
    m.add_function(wrap_pyfunction!(cavity::compute_cavity_volume, m)?)?;
    m.add_function(wrap_pyfunction!(cavity::find_interior_residues, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::kabsch_rmsd, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::superpose, m)?)?;
    Ok(())
}
