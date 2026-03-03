//! KD-tree based spatial queries — clash detection and contact finding.

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use kiddo::KdTree;
use kiddo::SquaredEuclidean;

/// Check for steric clashes between two sets of coordinates.
///
/// Parameters
/// ----------
/// coords_a : numpy.ndarray
///     Shape (N, 3) — first set of atom coordinates.
/// coords_b : numpy.ndarray
///     Shape (M, 3) — second set of atom coordinates.
/// cutoff : float
///     Distance cutoff in Angstroms. Pairs closer than this are clashes.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape (K, 3) array where each row is [index_a, index_b, distance].
///     Only pairs with distance < cutoff are returned.
#[pyfunction]
#[pyo3(signature = (coords_a, coords_b, cutoff=2.0))]
pub fn clash_check<'py>(
    py: Python<'py>,
    coords_a: PyReadonlyArray2<'py, f64>,
    coords_b: PyReadonlyArray2<'py, f64>,
    cutoff: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = coords_a.as_array();
    let b = coords_b.as_array();
    let n_a = a.shape()[0];
    let n_b = b.shape()[0];

    // Build KD-tree from set B
    let mut tree: KdTree<f64, 3> = KdTree::new();
    for i in 0..n_b {
        tree.add(&[b[[i, 0]], b[[i, 1]], b[[i, 2]]], i as u64);
    }

    let cutoff_sq = cutoff * cutoff;
    let mut clashes: Vec<[f64; 3]> = Vec::new();

    for i in 0..n_a {
        let query = [a[[i, 0]], a[[i, 1]], a[[i, 2]]];
        let neighbours = tree.within::<SquaredEuclidean>(&query, cutoff_sq);

        for nb in neighbours {
            let dist = nb.distance.sqrt();
            clashes.push([i as f64, nb.item as f64, dist]);
        }
    }

    let n_clashes = clashes.len();
    if n_clashes == 0 {
        let empty = PyArray2::from_vec(py, &vec![0.0f64; 0])
            .reshape([0, 3])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        return Ok(empty.to_owned());
    }

    let mut flat = vec![0.0f64; n_clashes * 3];
    for (i, c) in clashes.iter().enumerate() {
        flat[i * 3] = c[0];
        flat[i * 3 + 1] = c[1];
        flat[i * 3 + 2] = c[2];
    }

    let result = PyArray2::from_vec(py, &flat)
        .reshape([n_clashes, 3])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(result.to_owned())
}

/// Find all contacts between two sets of coordinates within a cutoff distance.
///
/// Similar to clash_check but intended for interface analysis with a larger cutoff.
///
/// Parameters
/// ----------
/// coords_a : numpy.ndarray
///     Shape (N, 3) atom coordinates.
/// coords_b : numpy.ndarray
///     Shape (M, 3) atom coordinates.
/// cutoff : float
///     Contact distance cutoff (Å). Default 5.0 for interface contacts.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape (K, 3) — [index_a, index_b, distance] for each contact.
#[pyfunction]
#[pyo3(signature = (coords_a, coords_b, cutoff=5.0))]
pub fn find_contacts<'py>(
    py: Python<'py>,
    coords_a: PyReadonlyArray2<'py, f64>,
    coords_b: PyReadonlyArray2<'py, f64>,
    cutoff: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // Reuse clash_check with a different default cutoff
    clash_check(py, coords_a, coords_b, cutoff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kdtree_basic() {
        let mut tree: KdTree<f64, 3> = KdTree::new();
        tree.add(&[0.0, 0.0, 0.0], 0);
        tree.add(&[1.0, 0.0, 0.0], 1);
        tree.add(&[10.0, 0.0, 0.0], 2);

        let neighbours = tree.within::<SquaredEuclidean>(&[0.0, 0.0, 0.0], 4.0); // cutoff=2.0 → sq=4.0
        assert_eq!(neighbours.len(), 2); // points 0 and 1
    }
}
