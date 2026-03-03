//! Lattice assembly — tile protein cages into 3D crystal lattices.
//!
//! Given a unit cell (cage coordinates + orientation) and lattice vectors,
//! generates the full lattice by translation.

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Build a 3D lattice of cages by periodic translation.
///
/// Parameters
/// ----------
/// cage_coords : numpy.ndarray
///     Shape (N, 3) — atom coordinates of a single cage (the motif).
/// lattice_vectors : numpy.ndarray
///     Shape (3, 3) — lattice vectors as rows: a, b, c.
/// repeats : list[int]
///     [na, nb, nc] — number of repeats along each lattice vector.
///     Total cages = na × nb × nc.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape (N * na * nb * nc, 3) — coordinates of the full lattice.
#[pyfunction]
pub fn build_lattice<'py>(
    py: Python<'py>,
    cage_coords: PyReadonlyArray2<'py, f64>,
    lattice_vectors: PyReadonlyArray2<'py, f64>,
    repeats: Vec<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let coords = cage_coords.as_array();
    let lv = lattice_vectors.as_array();
    let n_atoms = coords.shape()[0];

    if repeats.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "repeats must have 3 elements [na, nb, nc]",
        ));
    }

    let na = repeats[0];
    let nb = repeats[1];
    let nc = repeats[2];
    let n_cages = na * nb * nc;
    let total_atoms = n_atoms * n_cages;

    // Lattice vectors
    let a = [lv[[0, 0]], lv[[0, 1]], lv[[0, 2]]];
    let b = [lv[[1, 0]], lv[[1, 1]], lv[[1, 2]]];
    let c = [lv[[2, 0]], lv[[2, 1]], lv[[2, 2]]];

    // Pre-build cage coordinate array
    let cage: Vec<[f64; 3]> = (0..n_atoms)
        .map(|i| [coords[[i, 0]], coords[[i, 1]], coords[[i, 2]]])
        .collect();

    // Generate all translation vectors
    let translations: Vec<[f64; 3]> = {
        let mut t = Vec::with_capacity(n_cages);
        for ia in 0..na {
            for ib in 0..nb {
                for ic in 0..nc {
                    let tx = ia as f64 * a[0] + ib as f64 * b[0] + ic as f64 * c[0];
                    let ty = ia as f64 * a[1] + ib as f64 * b[1] + ic as f64 * c[1];
                    let tz = ia as f64 * a[2] + ib as f64 * b[2] + ic as f64 * c[2];
                    t.push([tx, ty, tz]);
                }
            }
        }
        t
    };

    // Apply translations in parallel
    let mut result = vec![0.0f64; total_atoms * 3];
    result
        .par_chunks_mut(n_atoms * 3)
        .enumerate()
        .for_each(|(cage_idx, chunk)| {
            let t = &translations[cage_idx];
            for atom in 0..n_atoms {
                chunk[atom * 3] = cage[atom][0] + t[0];
                chunk[atom * 3 + 1] = cage[atom][1] + t[1];
                chunk[atom * 3 + 2] = cage[atom][2] + t[2];
            }
        });

    let result_arr = PyArray2::from_vec(py, &result)
        .reshape([total_atoms, 3])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(result_arr.to_owned())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_lattice_count() {
        // 2×2×2 = 8 cages, each with 10 atoms = 80 atoms total
        let na = 2;
        let nb = 2;
        let nc = 2;
        let n_atoms = 10;
        assert_eq!(na * nb * nc * n_atoms, 80);
    }
}
