//! Solvent-Accessible Surface Area (SASA) computation.
//!
//! Implements the Shrake-Rupley algorithm with configurable number of
//! test points per atom. Parallelised with rayon.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Probe radius for water (Å).
const PROBE_RADIUS: f64 = 1.4;

/// Generate uniformly distributed points on a unit sphere using the
/// golden spiral method.
fn sphere_points(n: usize) -> Vec<[f64; 3]> {
    let mut points = Vec::with_capacity(n);
    let golden_angle = PI * (3.0 - 5.0_f64.sqrt());

    for i in 0..n {
        let y = 1.0 - 2.0 * (i as f64) / ((n - 1) as f64);
        let radius = (1.0 - y * y).sqrt();
        let theta = golden_angle * i as f64;
        points.push([radius * theta.cos(), y, radius * theta.sin()]);
    }
    points
}

/// Compute per-atom SASA using the Shrake-Rupley algorithm.
///
/// Parameters
/// ----------
/// coords : numpy.ndarray
///     Shape (N, 3) — atom coordinates.
/// radii : numpy.ndarray
///     Shape (N,) — van der Waals radii.
/// n_points : int
///     Number of test points per atom sphere. Default 92.
/// probe_radius : float
///     Solvent probe radius (Å). Default 1.4.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape (N,) — SASA per atom in Å².
#[pyfunction]
#[pyo3(signature = (coords, radii, n_points=92, probe_radius=1.4))]
pub fn compute_sasa<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    radii: PyReadonlyArray1<'py, f64>,
    n_points: usize,
    probe_radius: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = coords.as_array();
    let r = radii.as_array();
    let n_atoms = c.shape()[0];

    let test_points = sphere_points(n_points);

    // Pre-build coordinate and radius arrays for thread safety
    let atom_coords: Vec<[f64; 3]> = (0..n_atoms)
        .map(|i| [c[[i, 0]], c[[i, 1]], c[[i, 2]]])
        .collect();
    let atom_radii: Vec<f64> = (0..n_atoms).map(|i| r[i]).collect();

    // For each atom, compute the fraction of test points not buried by other atoms
    let sasa: Vec<f64> = (0..n_atoms)
        .into_par_iter()
        .map(|i| {
            let ri = atom_radii[i] + probe_radius;
            let ri_sq_area = 4.0 * PI * ri * ri;
            let mut accessible = 0usize;

            for tp in &test_points {
                let px = atom_coords[i][0] + ri * tp[0];
                let py = atom_coords[i][1] + ri * tp[1];
                let pz = atom_coords[i][2] + ri * tp[2];

                let mut buried = false;
                for j in 0..n_atoms {
                    if j == i {
                        continue;
                    }
                    let rj = atom_radii[j] + probe_radius;
                    let dx = px - atom_coords[j][0];
                    let dy = py - atom_coords[j][1];
                    let dz = pz - atom_coords[j][2];
                    let dist_sq = dx * dx + dy * dy + dz * dz;
                    if dist_sq < rj * rj {
                        buried = true;
                        break;
                    }
                }
                if !buried {
                    accessible += 1;
                }
            }

            ri_sq_area * (accessible as f64) / (n_points as f64)
        })
        .collect();

    Ok(PyArray1::from_vec(py, &sasa).to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_points_count() {
        let pts = sphere_points(100);
        assert_eq!(pts.len(), 100);
    }

    #[test]
    fn test_sphere_points_on_unit_sphere() {
        let pts = sphere_points(50);
        for p in &pts {
            let dist = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((dist - 1.0).abs() < 1e-6, "Point not on unit sphere");
        }
    }
}
