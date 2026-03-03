//! Symmetry operations for polyhedral point groups (T, O, I).
//!
//! Provides generation, storage, and application of 3×3 rotation matrices
//! for the chiral tetrahedral (12 ops), octahedral (24 ops), and icosahedral
//! (60 ops) point groups used in protein cage design.

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Identity matrix.
const I3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Build a rotation matrix for angle `theta` (radians) about axis (ux, uy, uz).
/// Axis must be unit-length.
fn rotation_matrix(ux: f64, uy: f64, uz: f64, theta: f64) -> [[f64; 3]; 3] {
    let c = theta.cos();
    let s = theta.sin();
    let t = 1.0 - c;
    [
        [t * ux * ux + c, t * ux * uy - s * uz, t * ux * uz + s * uy],
        [t * ux * uy + s * uz, t * uy * uy + c, t * uy * uz - s * ux],
        [t * ux * uz - s * uy, t * uy * uz + s * ux, t * uz * uz + c],
    ]
}

/// Multiply two 3×3 matrices.
fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// Check if two rotation matrices are approximately equal.
fn mat_approx_eq(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3], tol: f64) -> bool {
    for i in 0..3 {
        for j in 0..3 {
            if (a[i][j] - b[i][j]).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Generate full group by closure from a set of generators.
fn generate_group(generators: &[[[f64; 3]; 3]], max_ops: usize) -> Vec<[[f64; 3]; 3]> {
    let tol = 1e-10;
    let mut group: Vec<[[f64; 3]; 3]> = vec![I3];

    // Add generators
    for g in generators {
        let already = group.iter().any(|m| mat_approx_eq(m, g, tol));
        if !already {
            group.push(*g);
        }
    }

    // Close under multiplication
    let mut changed = true;
    while changed && group.len() < max_ops + 1 {
        changed = false;
        let n = group.len();
        let mut new_ops = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let product = mat_mul(&group[i], &group[j]);
                let already = group.iter().chain(new_ops.iter()).any(|m| mat_approx_eq(m, &product, tol));
                if !already {
                    new_ops.push(product);
                    changed = true;
                }
            }
        }
        group.extend(new_ops);
    }

    group
}

/// Golden ratio.
const PHI: f64 = 1.618033988749895;

/// Generate tetrahedral group T (12 operations).
///
/// Generators: C3 about (1,1,1) and C2 about (1,0,0).
fn tetrahedral_operations() -> Vec<[[f64; 3]; 3]> {
    let sqrt3_inv = 1.0 / 3.0_f64.sqrt();
    let c3 = rotation_matrix(sqrt3_inv, sqrt3_inv, sqrt3_inv, 2.0 * PI / 3.0);
    let c2 = rotation_matrix(1.0, 0.0, 0.0, PI);
    generate_group(&[c3, c2], 12)
}

/// Generate octahedral group O (24 operations).
///
/// Generators: C4 about z-axis and C3 about (1,1,1).
fn octahedral_operations() -> Vec<[[f64; 3]; 3]> {
    let c4 = rotation_matrix(0.0, 0.0, 1.0, PI / 2.0);
    let sqrt3_inv = 1.0 / 3.0_f64.sqrt();
    let c3 = rotation_matrix(sqrt3_inv, sqrt3_inv, sqrt3_inv, 2.0 * PI / 3.0);
    generate_group(&[c4, c3], 24)
}

/// Generate icosahedral group I (60 operations).
///
/// Generators: C5 about (0, 1, φ)/|(0,1,φ)| and C3 about (1,1,1)/√3.
fn icosahedral_operations() -> Vec<[[f64; 3]; 3]> {
    // C5 axis along (0, 1, φ)
    let norm = (1.0 + PHI * PHI).sqrt();
    let c5 = rotation_matrix(0.0, 1.0 / norm, PHI / norm, 2.0 * PI / 5.0);
    // C3 axis along (1, 1, 1)
    let sqrt3_inv = 1.0 / 3.0_f64.sqrt();
    let c3 = rotation_matrix(sqrt3_inv, sqrt3_inv, sqrt3_inv, 2.0 * PI / 3.0);
    // C2 axis along (0, 0, 1) — we need an additional generator to reach 60
    let c2 = rotation_matrix(0.0, 0.0, 1.0, PI);
    generate_group(&[c5, c3, c2], 60)
}

/// Get all rotation matrices for a symmetry group.
///
/// Parameters
/// ----------
/// group_name : str
///     One of "T" (tetrahedral, 12 ops), "O" (octahedral, 24 ops),
///     "I" (icosahedral, 60 ops).
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape (N*3, 3) array of stacked 3×3 rotation matrices,
///     where N is the number of group operations.
#[pyfunction]
pub fn get_symmetry_operations<'py>(
    py: Python<'py>,
    group_name: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ops = match group_name {
        "T" | "tetrahedral" => tetrahedral_operations(),
        "O" | "octahedral" => octahedral_operations(),
        "I" | "icosahedral" => icosahedral_operations(),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown symmetry group '{}'. Expected 'T', 'O', or 'I'.",
                group_name
            )))
        }
    };

    let n = ops.len();
    let mut flat = vec![0.0f64; n * 3 * 3];
    for (k, op) in ops.iter().enumerate() {
        for i in 0..3 {
            for j in 0..3 {
                flat[k * 9 + i * 3 + j] = op[i][j];
            }
        }
    }

    let result = PyArray2::from_vec(py, &flat)
        .reshape([n * 3, 3])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(result.to_owned())
}

/// Get minimal generator set for a symmetry group.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape (G*3, 3) stacked generator matrices.
#[pyfunction]
pub fn get_group_generators<'py>(
    py: Python<'py>,
    group_name: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let generators: Vec<[[f64; 3]; 3]> = match group_name {
        "T" | "tetrahedral" => {
            let sqrt3_inv = 1.0 / 3.0_f64.sqrt();
            vec![
                rotation_matrix(sqrt3_inv, sqrt3_inv, sqrt3_inv, 2.0 * PI / 3.0),
                rotation_matrix(1.0, 0.0, 0.0, PI),
            ]
        }
        "O" | "octahedral" => {
            let sqrt3_inv = 1.0 / 3.0_f64.sqrt();
            vec![
                rotation_matrix(0.0, 0.0, 1.0, PI / 2.0),
                rotation_matrix(sqrt3_inv, sqrt3_inv, sqrt3_inv, 2.0 * PI / 3.0),
            ]
        }
        "I" | "icosahedral" => {
            let norm = (1.0 + PHI * PHI).sqrt();
            let sqrt3_inv = 1.0 / 3.0_f64.sqrt();
            vec![
                rotation_matrix(0.0, 1.0 / norm, PHI / norm, 2.0 * PI / 5.0),
                rotation_matrix(sqrt3_inv, sqrt3_inv, sqrt3_inv, 2.0 * PI / 3.0),
                rotation_matrix(0.0, 0.0, 1.0, PI),
            ]
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown symmetry group '{}'.", group_name
            )))
        }
    };

    let n = generators.len();
    let mut flat = vec![0.0f64; n * 9];
    for (k, g) in generators.iter().enumerate() {
        for i in 0..3 {
            for j in 0..3 {
                flat[k * 9 + i * 3 + j] = g[i][j];
            }
        }
    }

    let result = PyArray2::from_vec(py, &flat)
        .reshape([n * 3, 3])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(result.to_owned())
}

/// Apply symmetry operations to a set of atomic coordinates.
///
/// Parameters
/// ----------
/// coords : numpy.ndarray
///     Shape (N, 3) — coordinates of atoms in the asymmetric unit.
/// rotations : numpy.ndarray
///     Shape (M*3, 3) — stacked 3×3 rotation matrices from `get_symmetry_operations`.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape (N*M, 3) — coordinates after applying all M symmetry operations.
///     First N rows are the identity copy, next N are the second operation, etc.
#[pyfunction]
pub fn apply_symmetry_ops<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    rotations: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let coords = coords.as_array();
    let rots = rotations.as_array();

    let n_atoms = coords.shape()[0];
    let n_rot_rows = rots.shape()[0];
    if n_rot_rows % 3 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rotations must have shape (M*3, 3), where M is the number of operations.",
        ));
    }
    let n_ops = n_rot_rows / 3;
    let total = n_atoms * n_ops;

    // Parallel expansion using rayon
    let mut result = vec![0.0f64; total * 3];

    // Process each operation in parallel
    result
        .par_chunks_mut(n_atoms * 3)
        .enumerate()
        .for_each(|(op_idx, chunk)| {
            let r_off = op_idx * 3;
            for atom in 0..n_atoms {
                let x = coords[[atom, 0]];
                let y = coords[[atom, 1]];
                let z = coords[[atom, 2]];
                chunk[atom * 3] = rots[[r_off, 0]] * x + rots[[r_off, 1]] * y + rots[[r_off, 2]] * z;
                chunk[atom * 3 + 1] =
                    rots[[r_off + 1, 0]] * x + rots[[r_off + 1, 1]] * y + rots[[r_off + 1, 2]] * z;
                chunk[atom * 3 + 2] =
                    rots[[r_off + 2, 0]] * x + rots[[r_off + 2, 1]] * y + rots[[r_off + 2, 2]] * z;
            }
        });

    let result_arr = PyArray2::from_vec(py, &result)
        .reshape([total, 3])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(result_arr.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tetrahedral_group_size() {
        let ops = tetrahedral_operations();
        assert_eq!(ops.len(), 12, "Tetrahedral group should have 12 operations");
    }

    #[test]
    fn test_octahedral_group_size() {
        let ops = octahedral_operations();
        assert_eq!(ops.len(), 24, "Octahedral group should have 24 operations");
    }

    #[test]
    fn test_icosahedral_group_size() {
        let ops = icosahedral_operations();
        assert_eq!(ops.len(), 60, "Icosahedral group should have 60 operations");
    }

    #[test]
    fn test_identity_present() {
        for group_fn in [tetrahedral_operations, octahedral_operations, icosahedral_operations] {
            let ops = group_fn();
            let has_identity = ops.iter().any(|m| mat_approx_eq(m, &I3, 1e-10));
            assert!(has_identity, "Group must contain the identity");
        }
    }

    #[test]
    fn test_closure() {
        // Every product of two group elements must be in the group
        let ops = tetrahedral_operations();
        for a in &ops {
            for b in &ops {
                let prod = mat_mul(a, b);
                let found = ops.iter().any(|m| mat_approx_eq(m, &prod, 1e-8));
                assert!(found, "Group is not closed under multiplication");
            }
        }
    }

    #[test]
    fn test_rotation_matrices_are_orthogonal() {
        let ops = icosahedral_operations();
        for op in &ops {
            // R^T R = I
            let rt_r = mat_mul(
                &[
                    [op[0][0], op[1][0], op[2][0]],
                    [op[0][1], op[1][1], op[2][1]],
                    [op[0][2], op[1][2], op[2][2]],
                ],
                op,
            );
            assert!(
                mat_approx_eq(&rt_r, &I3, 1e-10),
                "Rotation matrix must be orthogonal"
            );
        }
    }
}
