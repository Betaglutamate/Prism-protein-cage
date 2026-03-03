//! Platonic solid geometry — vertex and face generation.
//!
//! Provides canonical vertex coordinates and face connectivity for the five
//! Platonic solids: tetrahedron, cube, octahedron, dodecahedron, icosahedron.
//! Used for cage wireframe visualisation, subunit placement, and cavity shape
//! reference.

use numpy::PyArray2;
use pyo3::prelude::*;
use std::f64::consts::PI;

/// Golden ratio φ = (1 + √5) / 2.
const PHI: f64 = 1.618033988749895;

/// Return vertices of a Platonic solid centred at the origin with unit
/// circumradius (all vertices at distance 1 from origin).
fn platonic_vertices(name: &str) -> Result<Vec<[f64; 3]>, String> {
    match name {
        "tetrahedron" => {
            // Regular tetrahedron inscribed in a cube
            let a = 1.0 / 3.0_f64.sqrt();
            Ok(vec![
                [a, a, a],
                [a, -a, -a],
                [-a, a, -a],
                [-a, -a, a],
            ])
        }
        "cube" | "hexahedron" => {
            let a = 1.0 / 3.0_f64.sqrt();
            Ok(vec![
                [a, a, a], [a, a, -a], [a, -a, a], [a, -a, -a],
                [-a, a, a], [-a, a, -a], [-a, -a, a], [-a, -a, -a],
            ])
        }
        "octahedron" => {
            Ok(vec![
                [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
            ])
        }
        "icosahedron" => {
            // 12 vertices: permutations of (0, ±1, ±φ) normalized
            let norm = (1.0 + PHI * PHI).sqrt();
            let a = 1.0 / norm;
            let b = PHI / norm;
            Ok(vec![
                [0.0, a, b], [0.0, a, -b], [0.0, -a, b], [0.0, -a, -b],
                [a, b, 0.0], [a, -b, 0.0], [-a, b, 0.0], [-a, -b, 0.0],
                [b, 0.0, a], [b, 0.0, -a], [-b, 0.0, a], [-b, 0.0, -a],
            ])
        }
        "dodecahedron" => {
            // 20 vertices from cube (±1,±1,±1) + rectangles (0,±1/φ,±φ) etc.
            let norm_cube = 3.0_f64.sqrt();
            let a = 1.0 / norm_cube;

            let inv_phi = 1.0 / PHI;
            let norm_rect = (1.0 + PHI * PHI).sqrt(); // same as icosa

            let mut verts = Vec::with_capacity(20);
            // 8 cube vertices
            for &sx in &[1.0, -1.0] {
                for &sy in &[1.0, -1.0] {
                    for &sz in &[1.0, -1.0] {
                        verts.push([sx * a, sy * a, sz * a]);
                    }
                }
            }
            // 12 rectangle vertices: (0, ±1/φ, ±φ) and cyclic permutations
            // We need to normalise to unit circumradius
            let rect_norm = (inv_phi * inv_phi + PHI * PHI).sqrt();
            let p = inv_phi / rect_norm;
            let q = PHI / rect_norm;
            for &s1 in &[1.0, -1.0] {
                for &s2 in &[1.0, -1.0] {
                    verts.push([0.0, s1 * p, s2 * q]);
                    verts.push([s1 * p, s2 * q, 0.0]);
                    verts.push([s2 * q, 0.0, s1 * p]);
                }
            }
            // Normalise all to unit circumradius
            for v in &mut verts {
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                v[0] /= len;
                v[1] /= len;
                v[2] /= len;
            }
            Ok(verts)
        }
        _ => Err(format!(
            "Unknown polyhedron '{}'. Expected: tetrahedron, cube, octahedron, \
             icosahedron, dodecahedron.",
            name
        )),
    }
}

/// Face connectivity for Platonic solids (vertex indices, 0-based).
fn platonic_faces(name: &str) -> Result<Vec<Vec<usize>>, String> {
    match name {
        "tetrahedron" => Ok(vec![
            vec![0, 1, 2],
            vec![0, 1, 3],
            vec![0, 2, 3],
            vec![1, 2, 3],
        ]),
        "cube" | "hexahedron" => Ok(vec![
            vec![0, 1, 3, 2], // +x
            vec![4, 5, 7, 6], // -x
            vec![0, 1, 5, 4], // +y
            vec![2, 3, 7, 6], // -y
            vec![0, 2, 6, 4], // +z
            vec![1, 3, 7, 5], // -z
        ]),
        "octahedron" => Ok(vec![
            vec![0, 2, 4], vec![0, 2, 5], vec![0, 3, 4], vec![0, 3, 5],
            vec![1, 2, 4], vec![1, 2, 5], vec![1, 3, 4], vec![1, 3, 5],
        ]),
        "icosahedron" => {
            // 20 triangular faces — computed from vertex adjacency
            // Using the standard icosahedron face table
            Ok(vec![
                vec![0, 2, 8],  vec![0, 4, 8],  vec![0, 4, 6],  vec![0, 6, 10],
                vec![0, 2, 10], vec![1, 3, 9],  vec![1, 4, 9],  vec![1, 4, 6],
                vec![1, 6, 11], vec![1, 3, 11], vec![2, 5, 8],  vec![2, 7, 10],
                vec![2, 5, 7],  vec![3, 5, 9],  vec![3, 7, 11], vec![3, 5, 7],
                vec![8, 5, 9],  vec![8, 4, 9],  vec![10, 6, 11],vec![10, 7, 11],
            ])
        }
        "dodecahedron" => {
            // 12 pentagonal faces — placeholder, exact connectivity depends on
            // vertex ordering. Return empty for now, compute from distances.
            Ok(vec![])
        }
        _ => Err(format!("Unknown polyhedron '{}'.", name)),
    }
}

/// Get vertex coordinates of a Platonic solid.
///
/// Parameters
/// ----------
/// name : str
///     One of "tetrahedron", "cube", "octahedron", "icosahedron", "dodecahedron".
/// scale : float
///     Circumradius in Angstroms. Default 1.0.
///
/// Returns
/// -------
/// numpy.ndarray
///     Shape (V, 3) vertex coordinates.
#[pyfunction]
#[pyo3(signature = (name, scale=1.0))]
pub fn get_polyhedron_vertices<'py>(
    py: Python<'py>,
    name: &str,
    scale: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let verts = platonic_vertices(name)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    let n = verts.len();
    let mut flat = vec![0.0f64; n * 3];
    for (i, v) in verts.iter().enumerate() {
        flat[i * 3] = v[0] * scale;
        flat[i * 3 + 1] = v[1] * scale;
        flat[i * 3 + 2] = v[2] * scale;
    }

    let result = PyArray2::from_vec(py, &flat)
        .reshape([n, 3])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(result.to_owned())
}

/// Get face connectivity of a Platonic solid.
///
/// Returns
/// -------
/// list[list[int]]
///     Each inner list contains vertex indices (0-based) forming a face.
#[pyfunction]
pub fn get_polyhedron_faces(name: &str) -> PyResult<Vec<Vec<usize>>> {
    platonic_faces(name).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tetrahedron_vertices() {
        let verts = platonic_vertices("tetrahedron").unwrap();
        assert_eq!(verts.len(), 4);
        // All vertices should be at distance 1 from origin
        for v in &verts {
            let dist = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((dist - 1.0).abs() < 1e-10, "Vertex not on unit sphere");
        }
    }

    #[test]
    fn test_icosahedron_vertices() {
        let verts = platonic_vertices("icosahedron").unwrap();
        assert_eq!(verts.len(), 12);
        for v in &verts {
            let dist = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((dist - 1.0).abs() < 1e-10, "Vertex not on unit sphere");
        }
    }

    #[test]
    fn test_octahedron_faces() {
        let faces = platonic_faces("octahedron").unwrap();
        assert_eq!(faces.len(), 8, "Octahedron has 8 faces");
    }
}
