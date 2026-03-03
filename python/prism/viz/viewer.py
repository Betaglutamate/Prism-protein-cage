"""
3D visualization for PRISM protein cages using py3Dmol.

Provides ``CageViewer`` — an interactive viewer for Jupyter notebooks
that can render subunits, full cages, cavity surfaces, symmetry axes,
interior residues, and lattice assemblies.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def _get_viewer(width: int = 800, height: int = 600):
    """Create a py3Dmol viewer, raising helpful error if missing."""
    try:
        import py3Dmol
    except ImportError:
        raise ImportError(
            "py3Dmol is required for 3D visualization. "
            "Install with: pip install py3Dmol"
        )
    return py3Dmol.view(width=width, height=height)


# ── Colour palettes ──────────────────────────────────────────────────

CHAIN_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939",
    "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39",
    "#e7ba52", "#e7cb94", "#843c39", "#ad494a", "#d6616b",
    "#e7969c", "#7b4173", "#a55194", "#ce6dbd", "#de9ed6",
    "#3182bd", "#6baed6", "#9ecae1", "#c6dbef", "#e6550d",
    "#fd8d3c", "#fdae6b", "#fdd0a2", "#31a354", "#74c476",
    "#a1d99b", "#c7e9c0", "#756bb1", "#9e9ac8", "#bcbddc",
    "#dadaec", "#636363", "#969696", "#bdbdbd", "#d9d9d9",
]


class CageViewer:
    """Interactive 3D cage viewer for Jupyter notebooks.

    Usage::

        from prism.viz.viewer import CageViewer
        v = CageViewer()
        v.show_subunit(pdb_path="subunit.pdb")
        v.show()
    """

    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self._viewer = None
        self._structures: list[dict] = []

    def _ensure_viewer(self):
        if self._viewer is None:
            self._viewer = _get_viewer(self.width, self.height)
        return self._viewer

    # ── Structure loading ─────────────────────────────────────────────

    def add_pdb(
        self,
        pdb_path: str,
        style: str = "cartoon",
        colour: Optional[str] = None,
        opacity: float = 1.0,
        label: Optional[str] = None,
    ) -> "CageViewer":
        """Add a PDB structure to the viewer.

        Parameters
        ----------
        pdb_path : str
            Path to PDB file.
        style : str
            Rendering style: cartoon, stick, sphere, surface, line.
        colour : str, optional
            Colour hex code. If None, coloured by chain.
        opacity : float
            Surface/cartoon opacity.
        label : str, optional
            Label for the structure.
        """
        with open(pdb_path) as f:
            pdb_data = f.read()

        self._structures.append({
            "data": pdb_data,
            "format": "pdb",
            "style": style,
            "colour": colour,
            "opacity": opacity,
            "label": label,
        })
        return self

    def show_subunit(
        self,
        pdb_path: Optional[str] = None,
        pdb_string: Optional[str] = None,
        colour: str = "#1f77b4",
        style: str = "cartoon",
    ) -> "CageViewer":
        """Show a single subunit."""
        v = self._ensure_viewer()
        data = pdb_string or open(pdb_path).read()
        v.addModel(data, "pdb")
        v.setStyle({"model": -1}, {style: {"color": colour}})
        return self

    def show_full_cage(
        self,
        pdb_path: Optional[str] = None,
        pdb_string: Optional[str] = None,
        colour_by_chain: bool = True,
        style: str = "cartoon",
    ) -> "CageViewer":
        """Show the full assembled cage, optionally colouring by chain."""
        v = self._ensure_viewer()
        data = pdb_string or open(pdb_path).read()
        v.addModel(data, "pdb")

        if colour_by_chain:
            # Parse chain IDs to assign colours
            chains = set()
            for line in data.split("\n"):
                if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
                    chains.add(line[21])
            for i, ch in enumerate(sorted(chains)):
                col = CHAIN_COLOURS[i % len(CHAIN_COLOURS)]
                v.setStyle(
                    {"model": -1, "chain": ch},
                    {style: {"color": col}},
                )
        else:
            v.setStyle({"model": -1}, {style: {}})
        return self

    def show_cavity(
        self,
        pdb_path: Optional[str] = None,
        pdb_string: Optional[str] = None,
        cavity_colour: str = "#ff6666",
        cavity_opacity: float = 0.3,
        cage_style: str = "cartoon",
        cage_opacity: float = 0.5,
    ) -> "CageViewer":
        """Show cage with translucent surface highlighting the cavity."""
        v = self._ensure_viewer()
        data = pdb_string or open(pdb_path).read()
        v.addModel(data, "pdb")
        v.setStyle({"model": -1}, {cage_style: {"opacity": cage_opacity}})
        v.addSurface(
            "VDW",
            {"opacity": cavity_opacity, "color": cavity_colour},
            {"model": -1},
        )
        return self

    def show_symmetry_axes(
        self,
        symmetry_group: str = "T",
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        axis_length: float = 30.0,
        axis_radius: float = 0.3,
    ) -> "CageViewer":
        """Overlay symmetry axes on the current view.

        Parameters
        ----------
        symmetry_group : str
            One of 'T', 'O', 'I'.
        center : tuple
            Centre of cage.
        axis_length : float
            Half-length of axis cylinders in Å.
        axis_radius : float
            Cylinder radius in Å.
        """
        from prism.core.symmetry import SymmetryGroup

        v = self._ensure_viewer()
        sg = SymmetryGroup.from_name(symmetry_group)
        axes = sg.get_symmetry_axes()

        # Colour by axis order
        axis_colours = {2: "#ff0000", 3: "#00cc00", 5: "#0000ff", 4: "#ffaa00"}

        for order, axis_list in axes.items():
            col = axis_colours.get(order, "#888888")
            for ax in axis_list:
                ax = np.asarray(ax, dtype=float)
                ax = ax / np.linalg.norm(ax)
                start = np.array(center) - axis_length * ax
                end = np.array(center) + axis_length * ax
                v.addCylinder({
                    "start": {"x": float(start[0]), "y": float(start[1]), "z": float(start[2])},
                    "end": {"x": float(end[0]), "y": float(end[1]), "z": float(end[2])},
                    "radius": axis_radius,
                    "fromCap": True,
                    "toCap": True,
                    "color": col,
                })
        return self

    def show_interior_residues(
        self,
        pdb_path: Optional[str] = None,
        pdb_string: Optional[str] = None,
        interior_indices: Optional[NDArray] = None,
        highlight_colour: str = "#ff4444",
        base_style: str = "cartoon",
        highlight_style: str = "stick",
    ) -> "CageViewer":
        """Show interior-facing residues as sticks against cartoon backbone."""
        v = self._ensure_viewer()
        data = pdb_string or open(pdb_path).read()
        v.addModel(data, "pdb")
        v.setStyle({"model": -1}, {base_style: {"color": "#cccccc", "opacity": 0.6}})

        if interior_indices is not None:
            for idx in interior_indices:
                v.setStyle(
                    {"model": -1, "resi": int(idx)},
                    {
                        highlight_style: {"color": highlight_colour},
                        base_style: {"color": "#cccccc", "opacity": 0.6},
                    },
                )
        return self

    def show_lattice(
        self,
        cage_coords_list: list[NDArray],
        cage_centers: list[NDArray],
        style: str = "sphere",
        sphere_radius: float = 2.0,
    ) -> "CageViewer":
        """Visualise lattice of cages as spheres at each cage centre.

        For a schematic view, the Cα positions are shown as small spheres
        and cage centres as larger spheres.
        """
        v = self._ensure_viewer()

        for i, (coords, center) in enumerate(zip(cage_coords_list, cage_centers)):
            col = CHAIN_COLOURS[i % len(CHAIN_COLOURS)]

            # Cage centre marker
            v.addSphere({
                "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
                "radius": sphere_radius,
                "color": col,
            })

            # Atoms — sample every Nth for performance
            step = max(1, len(coords) // 200)
            for c in coords[::step]:
                v.addSphere({
                    "center": {"x": float(c[0]), "y": float(c[1]), "z": float(c[2])},
                    "radius": 0.5,
                    "color": col,
                    "opacity": 0.4,
                })

        return self

    # ── Rendering ─────────────────────────────────────────────────────

    def add_label(
        self,
        text: str,
        position: tuple[float, float, float],
        colour: str = "#000000",
        font_size: int = 14,
    ) -> "CageViewer":
        """Add a text label at a 3D position."""
        v = self._ensure_viewer()
        v.addLabel(
            text,
            {
                "position": {"x": position[0], "y": position[1], "z": position[2]},
                "fontColor": colour,
                "fontSize": font_size,
                "backgroundColor": "white",
                "backgroundOpacity": 0.7,
            },
        )
        return self

    def zoom_to_fit(self) -> "CageViewer":
        """Zoom view to fit all structures."""
        v = self._ensure_viewer()
        v.zoomTo()
        return self

    def show(self) -> object:
        """Render the 3D viewer. Returns the py3Dmol view object."""
        v = self._ensure_viewer()

        # Apply any pending add_pdb structures
        for struct in self._structures:
            v.addModel(struct["data"], struct["format"])
            style_dict = {}
            style_name = struct["style"]
            style_opts = {}
            if struct["colour"]:
                style_opts["color"] = struct["colour"]
            if struct["opacity"] < 1.0:
                style_opts["opacity"] = struct["opacity"]
            style_dict[style_name] = style_opts
            v.setStyle({"model": -1}, style_dict)

        self._structures.clear()
        v.zoomTo()
        return v.show()

    def render_png(self) -> Optional[bytes]:
        """Return PNG screenshot bytes (requires IPython kernel)."""
        v = self._ensure_viewer()
        try:
            return v.png()
        except Exception:
            return None


# ── Convenience functions ─────────────────────────────────────────────

def quick_view(
    pdb_path: str,
    style: str = "cartoon",
    colour_by_chain: bool = True,
    width: int = 800,
    height: int = 600,
) -> object:
    """One-liner to view a PDB file.

    Returns the py3Dmol view object for display in Jupyter.
    """
    v = CageViewer(width, height)
    v.show_full_cage(pdb_path=pdb_path, colour_by_chain=colour_by_chain, style=style)
    return v.show()


def compare_designs(
    pdb_paths: list[str],
    labels: Optional[list[str]] = None,
    width: int = 1200,
    height: int = 400,
) -> object:
    """Show multiple designs side-by-side using a grid view.

    Each design gets its own column with labelled header.
    """
    try:
        import py3Dmol
    except ImportError:
        raise ImportError("py3Dmol is required. Install: pip install py3Dmol")

    n = len(pdb_paths)
    view = py3Dmol.view(
        width=width, height=height,
        linked=False,
        viewergrid=(1, n),
    )

    for i, path in enumerate(pdb_paths):
        with open(path) as f:
            data = f.read()
        view.addModel(data, "pdb", viewer=(0, i))
        view.setStyle({"model": -1}, {"cartoon": {}}, viewer=(0, i))

        if labels:
            view.addLabel(
                labels[i] if i < len(labels) else f"Design {i + 1}",
                {"position": {"x": 0, "y": 0, "z": 0}, "fontSize": 16},
                viewer=(0, i),
            )
        view.zoomTo(viewer=(0, i))

    return view.show()
