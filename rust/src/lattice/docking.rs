//! Interface contact enumeration for docking analysis.

use numpy::{PyArray2, PyReadonlyArray2, ndarray::Array2};
use pyo3::prelude::*;
use kiddo::KdTree;
use kiddo::SquaredEuclidean;

/// Enumerate residue-level contacts across a protein-protein interface.
///
/// Given two chains (by Cα coordinates), find all residue pairs within
/// a contact cutoff and return their indices and distances.
///
/// Parameters
/// ----------
/// ca_coords_a : numpy.ndarray
///     Shape (R1, 3) — Cα coordinates of chain A residues.
/// ca_coords_b : numpy.ndarray
///     Shape (R2, 3) — Cα coordinates of chain B residues.
/// cutoff : float
///     Contact distance cutoff (Å). Default 8.0 for Cα-Cα contacts.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape (K, 3) — [residue_index_a, residue_index_b, distance].
#[pyfunction]
#[pyo3(signature = (ca_coords_a, ca_coords_b, cutoff=8.0))]
pub fn enumerate_interface_contacts<'py>(
    py: Python<'py>,
    ca_coords_a: PyReadonlyArray2<'py, f64>,
    ca_coords_b: PyReadonlyArray2<'py, f64>,
    cutoff: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = ca_coords_a.as_array();
    let b = ca_coords_b.as_array();
    let n_a = a.shape()[0];
    let n_b = b.shape()[0];

    let mut tree: KdTree<f64, 3> = KdTree::new();
    for i in 0..n_b {
        tree.add(&[b[[i, 0]], b[[i, 1]], b[[i, 2]]], i as u64);
    }

    let cutoff_sq = cutoff * cutoff;
    let mut contacts: Vec<[f64; 3]> = Vec::new();

    for i in 0..n_a {
        let query = [a[[i, 0]], a[[i, 1]], a[[i, 2]]];
        let neighbours = tree.within::<SquaredEuclidean>(&query, cutoff_sq);

        for nb in neighbours {
            let dist = nb.distance.sqrt();
            contacts.push([i as f64, nb.item as f64, dist]);
        }
    }

    let n = contacts.len();
    if n == 0 {
        let arr = Array2::<f64>::zeros((0, 3));
        return Ok(PyArray2::from_owned_array_bound(py, arr));
    }

    let mut flat = vec![0.0f64; n * 3];
    for (i, c) in contacts.iter().enumerate() {
        flat[i * 3] = c[0];
        flat[i * 3 + 1] = c[1];
        flat[i * 3 + 2] = c[2];
    }

    let arr = Array2::from_shape_vec((n, 3), flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array_bound(py, arr))
}
