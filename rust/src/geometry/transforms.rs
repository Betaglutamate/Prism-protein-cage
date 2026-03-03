//! Coordinate transforms — RMSD, superposition (Kabsch algorithm), centering.

use numpy::{PyArray2, PyReadonlyArray2, ndarray::Array2};
use pyo3::prelude::*;

/// Compute centroid of an (N, 3) coordinate array.
fn centroid(coords: &[[f64; 3]]) -> [f64; 3] {
    let n = coords.len() as f64;
    let mut c = [0.0; 3];
    for p in coords {
        c[0] += p[0];
        c[1] += p[1];
        c[2] += p[2];
    }
    c[0] /= n;
    c[1] /= n;
    c[2] /= n;
    c
}

/// Kabsch algorithm: find the optimal rotation matrix to superpose `mobile`
/// onto `target` (both centred at origin).
///
/// Returns (rotation_3x3, rmsd).
fn kabsch(mobile: &[[f64; 3]], target: &[[f64; 3]]) -> ([[f64; 3]; 3], f64) {
    let n = mobile.len();
    assert_eq!(n, target.len());

    // Covariance matrix H = mobile^T @ target
    let mut h = [[0.0f64; 3]; 3];
    for k in 0..n {
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] += mobile[k][i] * target[k][j];
            }
        }
    }

    // SVD of H using a simple Jacobi-like approach
    // For production, we'd use nalgebra, but let's use it here
    let h_mat = nalgebra::Matrix3::new(
        h[0][0], h[0][1], h[0][2],
        h[1][0], h[1][1], h[1][2],
        h[2][0], h[2][1], h[2][2],
    );

    let svd = h_mat.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();

    // Ensure proper rotation (det = +1)
    let d = (v_t.transpose() * u.transpose()).determinant();
    let sign = if d < 0.0 { -1.0 } else { 1.0 };

    let diag = nalgebra::Matrix3::new(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, sign,
    );

    let rotation = v_t.transpose() * diag * u.transpose();

    // Compute RMSD
    let mut rmsd_sum = 0.0;
    for k in 0..n {
        let rx = rotation[(0, 0)] * mobile[k][0] + rotation[(0, 1)] * mobile[k][1] + rotation[(0, 2)] * mobile[k][2];
        let ry = rotation[(1, 0)] * mobile[k][0] + rotation[(1, 1)] * mobile[k][1] + rotation[(1, 2)] * mobile[k][2];
        let rz = rotation[(2, 0)] * mobile[k][0] + rotation[(2, 1)] * mobile[k][1] + rotation[(2, 2)] * mobile[k][2];
        rmsd_sum += (rx - target[k][0]).powi(2) + (ry - target[k][1]).powi(2) + (rz - target[k][2]).powi(2);
    }
    let rmsd = (rmsd_sum / n as f64).sqrt();

    let rot_arr = [
        [rotation[(0, 0)], rotation[(0, 1)], rotation[(0, 2)]],
        [rotation[(1, 0)], rotation[(1, 1)], rotation[(1, 2)]],
        [rotation[(2, 0)], rotation[(2, 1)], rotation[(2, 2)]],
    ];

    (rot_arr, rmsd)
}

/// Compute RMSD between two coordinate sets after optimal superposition.
///
/// Parameters
/// ----------
/// mobile : numpy.ndarray
///     Shape (N, 3) coordinates to be superposed.
/// target : numpy.ndarray
///     Shape (N, 3) reference coordinates.
///
/// Returns
/// -------
/// float
///     RMSD in Angstroms after optimal superposition.
#[pyfunction]
pub fn kabsch_rmsd<'py>(
    _py: Python<'py>,
    mobile: PyReadonlyArray2<'py, f64>,
    target: PyReadonlyArray2<'py, f64>,
) -> PyResult<f64> {
    let mob = mobile.as_array();
    let tgt = target.as_array();
    let n = mob.shape()[0];

    if n != tgt.shape()[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mobile and target must have the same number of atoms",
        ));
    }

    let mob_vec: Vec<[f64; 3]> = (0..n).map(|i| [mob[[i, 0]], mob[[i, 1]], mob[[i, 2]]]).collect();
    let tgt_vec: Vec<[f64; 3]> = (0..n).map(|i| [tgt[[i, 0]], tgt[[i, 1]], tgt[[i, 2]]]).collect();

    // Centre both
    let mob_c = centroid(&mob_vec);
    let tgt_c = centroid(&tgt_vec);
    let mob_centered: Vec<[f64; 3]> = mob_vec
        .iter()
        .map(|p| [p[0] - mob_c[0], p[1] - mob_c[1], p[2] - mob_c[2]])
        .collect();
    let tgt_centered: Vec<[f64; 3]> = tgt_vec
        .iter()
        .map(|p| [p[0] - tgt_c[0], p[1] - tgt_c[1], p[2] - tgt_c[2]])
        .collect();

    let (_, rmsd) = kabsch(&mob_centered, &tgt_centered);
    Ok(rmsd)
}

/// Superpose mobile coordinates onto target using the Kabsch algorithm.
///
/// Returns
/// -------
/// tuple
///     (superposed_coords: ndarray shape (N,3),
///      rotation: ndarray shape (3,3),
///      rmsd: float)
#[pyfunction]
pub fn superpose<'py>(
    py: Python<'py>,
    mobile: PyReadonlyArray2<'py, f64>,
    target: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, f64)> {
    let mob = mobile.as_array();
    let tgt = target.as_array();
    let n = mob.shape()[0];

    if n != tgt.shape()[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mobile and target must have the same number of atoms",
        ));
    }

    let mob_vec: Vec<[f64; 3]> = (0..n).map(|i| [mob[[i, 0]], mob[[i, 1]], mob[[i, 2]]]).collect();
    let tgt_vec: Vec<[f64; 3]> = (0..n).map(|i| [tgt[[i, 0]], tgt[[i, 1]], tgt[[i, 2]]]).collect();

    let mob_c = centroid(&mob_vec);
    let tgt_c = centroid(&tgt_vec);

    let mob_centered: Vec<[f64; 3]> = mob_vec
        .iter()
        .map(|p| [p[0] - mob_c[0], p[1] - mob_c[1], p[2] - mob_c[2]])
        .collect();
    let tgt_centered: Vec<[f64; 3]> = tgt_vec
        .iter()
        .map(|p| [p[0] - tgt_c[0], p[1] - tgt_c[1], p[2] - tgt_c[2]])
        .collect();

    let (rot, rmsd) = kabsch(&mob_centered, &tgt_centered);

    // Apply rotation and translate to target centroid
    let mut result = vec![0.0f64; n * 3];
    for i in 0..n {
        let x = mob_centered[i][0];
        let y = mob_centered[i][1];
        let z = mob_centered[i][2];
        result[i * 3] = rot[0][0] * x + rot[0][1] * y + rot[0][2] * z + tgt_c[0];
        result[i * 3 + 1] = rot[1][0] * x + rot[1][1] * y + rot[1][2] * z + tgt_c[1];
        result[i * 3 + 2] = rot[2][0] * x + rot[2][1] * y + rot[2][2] * z + tgt_c[2];
    }

    let coords_out = PyArray2::from_owned_array_bound(py,
        Array2::from_shape_vec((n, 3), result)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?);

    let mut rot_flat = vec![0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            rot_flat[i * 3 + j] = rot[i][j];
        }
    }
    let rot_out = PyArray2::from_owned_array_bound(py,
        Array2::from_shape_vec((3, 3), rot_flat)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?);

    Ok((coords_out, rot_out, rmsd))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_centroid() {
        let coords = vec![[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]];
        let c = centroid(&coords);
        assert!((c[0] - 2.0).abs() < 1e-10);
        assert!((c[1] - 3.0).abs() < 1e-10);
        assert!((c[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_kabsch_identity() {
        let coords = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let c = centroid(&coords);
        let centered: Vec<[f64; 3]> = coords
            .iter()
            .map(|p| [p[0] - c[0], p[1] - c[1], p[2] - c[2]])
            .collect();
        let (rot, rmsd) = kabsch(&centered, &centered);
        assert!(rmsd < 1e-10, "Self-superposition RMSD should be ~0");
        // Rotation should be close to identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (rot[i][j] - expected).abs() < 1e-10,
                    "Self-superposition rotation should be identity"
                );
            }
        }
    }
}
