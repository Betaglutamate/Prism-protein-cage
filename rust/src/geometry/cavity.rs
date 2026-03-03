//! Cavity analysis — volume computation, interior surface detection,
//! and interior-facing residue identification.
//!
//! Uses Monte Carlo integration to estimate cavity volume and identifies
//! residues whose Cα atoms face the interior of a protein cage.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::Rng;
use rayon::prelude::*;

/// Van der Waals radii for common elements (Å).
fn vdw_radius(element: u8) -> f64 {
    match element {
        // C
        6 => 1.70,
        // N
        7 => 1.55,
        // O
        8 => 1.52,
        // S
        16 => 1.80,
        // H
        1 => 1.20,
        // P
        15 => 1.80,
        // Fe
        26 => 1.94,
        _ => 1.70, // default to carbon-like
    }
}

/// Check if a point is inside any atom's van der Waals sphere.
fn point_inside_protein(
    point: &[f64; 3],
    coords: &[[f64; 3]],
    radii: &[f64],
) -> bool {
    for (atom, r) in coords.iter().zip(radii.iter()) {
        let dx = point[0] - atom[0];
        let dy = point[1] - atom[1];
        let dz = point[2] - atom[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        if dist_sq < r * r {
            return true;
        }
    }
    false
}

/// Compute cavity volume of a protein cage by Monte Carlo integration.
///
/// The algorithm places random points inside a bounding sphere centred at
/// `center`. Points that are (a) within `cage_radius` of the center and
/// (b) NOT inside any atom's vdW sphere are counted as cavity. The cavity
/// volume is the fraction of such points times the bounding sphere volume.
///
/// Parameters
/// ----------
/// coords : numpy.ndarray
///     Shape (N, 3) — all atom coordinates of the full cage assembly.
/// elements : numpy.ndarray
///     Shape (N,) — atomic numbers (6=C, 7=N, 8=O, 16=S, etc.).
/// center : list[float]
///     [x, y, z] centre of the cage (typically [0, 0, 0]).
/// cage_radius : float
///     Approximate radius of the cage interior (Å). Points beyond this
///     are not sampled.
/// n_samples : int
///     Number of Monte Carlo samples. More = more accurate. 1_000_000 recommended.
///
/// Returns
/// -------
/// dict
///     {"volume_angstrom3": float, "inscribed_radius": float,
///      "void_fraction": float, "n_inside": int, "n_total": int}
#[pyfunction]
#[pyo3(signature = (coords, elements, center, cage_radius, n_samples=1_000_000))]
pub fn compute_cavity_volume<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    elements: PyReadonlyArray1<'py, u8>,
    center: Vec<f64>,
    cage_radius: f64,
    n_samples: usize,
) -> PyResult<PyObject> {
    let coords_arr = coords.as_array();
    let elems_arr = elements.as_array();
    let n_atoms = coords_arr.shape()[0];

    // Build local arrays for thread-safe access
    let atom_coords: Vec<[f64; 3]> = (0..n_atoms)
        .map(|i| [coords_arr[[i, 0]], coords_arr[[i, 1]], coords_arr[[i, 2]]])
        .collect();
    let radii: Vec<f64> = (0..n_atoms).map(|i| vdw_radius(elems_arr[i])).collect();
    let cx = center[0];
    let cy = center[1];
    let cz = center[2];

    // Parallel Monte Carlo — each thread gets its own RNG
    let n_threads = rayon::current_num_threads().max(1);
    let samples_per_thread = n_samples / n_threads;

    let counts: Vec<(usize, usize, f64)> = (0..n_threads)
        .into_par_iter()
        .map(|_thread_id| {
            let mut rng = rand::thread_rng();
            let mut n_inside_cavity = 0usize;
            let mut n_in_sphere = 0usize;
            let mut min_wall_dist = f64::MAX;

            for _ in 0..samples_per_thread {
                // Random point in bounding cube, reject if outside sphere
                let x = cx + (rng.gen::<f64>() * 2.0 - 1.0) * cage_radius;
                let y = cy + (rng.gen::<f64>() * 2.0 - 1.0) * cage_radius;
                let z = cz + (rng.gen::<f64>() * 2.0 - 1.0) * cage_radius;

                let dx = x - cx;
                let dy = y - cy;
                let dz = z - cz;
                let dist_from_center = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist_from_center > cage_radius {
                    continue;
                }
                n_in_sphere += 1;

                let point = [x, y, z];
                if !point_inside_protein(&point, &atom_coords, &radii) {
                    n_inside_cavity += 1;

                    // Track minimum distance to any atom (for inscribed radius estimate)
                    let mut min_d = f64::MAX;
                    for (atom, r) in atom_coords.iter().zip(radii.iter()) {
                        let d = ((point[0] - atom[0]).powi(2)
                            + (point[1] - atom[1]).powi(2)
                            + (point[2] - atom[2]).powi(2))
                        .sqrt()
                            - r;
                        if d < min_d {
                            min_d = d;
                        }
                    }
                    if min_d < min_wall_dist {
                        min_wall_dist = min_d;
                    }
                }
            }
            (n_inside_cavity, n_in_sphere, min_wall_dist)
        })
        .collect();

    let total_inside: usize = counts.iter().map(|c| c.0).sum();
    let total_in_sphere: usize = counts.iter().map(|c| c.1).sum();
    let inscribed_r: f64 = counts
        .iter()
        .map(|c| c.2)
        .fold(f64::MAX, f64::min);

    let sphere_vol = 4.0 / 3.0 * std::f64::consts::PI * cage_radius.powi(3);
    let void_fraction = if total_in_sphere > 0 {
        total_inside as f64 / total_in_sphere as f64
    } else {
        0.0
    };
    let volume = sphere_vol * void_fraction;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("volume_angstrom3", volume)?;
    dict.set_item("inscribed_radius", inscribed_r)?;
    dict.set_item("void_fraction", void_fraction)?;
    dict.set_item("n_inside", total_inside)?;
    dict.set_item("n_total", total_in_sphere)?;
    Ok(dict.into_any().unbind())
}

/// Identify residues whose Cα atoms face the cage interior.
///
/// A residue is "interior-facing" if its Cα→centre vector has a positive
/// projection along the Cα→Cβ direction (i.e., the side chain points inward),
/// AND the Cα is closer to the centre than to the cage exterior.
///
/// Parameters
/// ----------
/// ca_coords : numpy.ndarray
///     Shape (R, 3) — Cα coordinates, one per residue.
/// cb_coords : numpy.ndarray
///     Shape (R, 3) — Cβ coordinates (use Cα for glycine).
/// center : list[float]
///     Cage centre coordinates.
/// max_distance : float
///     Maximum Cα-to-centre distance to qualify as interior (Å).
///
/// Returns
/// -------
/// numpy.ndarray
///     1D array of residue indices (0-based) that face the interior.
#[pyfunction]
pub fn find_interior_residues<'py>(
    py: Python<'py>,
    ca_coords: PyReadonlyArray2<'py, f64>,
    cb_coords: PyReadonlyArray2<'py, f64>,
    center: Vec<f64>,
    max_distance: f64,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let ca = ca_coords.as_array();
    let cb = cb_coords.as_array();
    let n_res = ca.shape()[0];

    let cx = center[0];
    let cy = center[1];
    let cz = center[2];

    let mut interior: Vec<u32> = Vec::new();

    for i in 0..n_res {
        let ca_x = ca[[i, 0]];
        let ca_y = ca[[i, 1]];
        let ca_z = ca[[i, 2]];

        // Vector from Cα to cage centre
        let to_center = [cx - ca_x, cy - ca_y, cz - ca_z];
        let dist_to_center = (to_center[0].powi(2) + to_center[1].powi(2) + to_center[2].powi(2)).sqrt();

        if dist_to_center > max_distance {
            continue;
        }

        // Vector from Cα to Cβ (side-chain direction)
        let ca_cb = [
            cb[[i, 0]] - ca_x,
            cb[[i, 1]] - ca_y,
            cb[[i, 2]] - ca_z,
        ];

        // If side chain points toward centre, residue is interior-facing
        let dot = to_center[0] * ca_cb[0] + to_center[1] * ca_cb[1] + to_center[2] * ca_cb[2];
        if dot > 0.0 {
            interior.push(i as u32);
        }
    }

    Ok(PyArray1::from_vec(py, &interior).to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vdw_radius() {
        assert!((vdw_radius(6) - 1.70).abs() < 1e-10);
        assert!((vdw_radius(7) - 1.55).abs() < 1e-10);
    }

    #[test]
    fn test_point_inside_sphere() {
        let coords = vec![[0.0, 0.0, 0.0]];
        let radii = vec![2.0];
        assert!(point_inside_protein(&[0.5, 0.5, 0.5], &coords, &radii));
        assert!(!point_inside_protein(&[3.0, 0.0, 0.0], &coords, &radii));
    }
}
