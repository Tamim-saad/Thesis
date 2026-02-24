"""
Optimized Surface Visualizer - Generated CDB Files
=====================================================
Reads generated_*.cdb files from thesis_output/ and exports
optimized .vtp surface meshes to output_analysis/optimized_cdb/

Usage:
  python _optimized_surface.py                    # Process all generated CDBs
  python _optimized_surface.py <file.cdb>         # Process a single file
  python _optimized_surface.py --interactive      # Process all + show last in 3D viewer
"""

import pyvista as pv
from ansys.mapdl import reader as pymapdl_reader
import os
import sys
import glob
import numpy as np


def process_mesh(input_file, output_dir):
    """Process a single CDB file: load, extract surface, save .vtp."""
    basename = os.path.basename(input_file)
    if not os.path.exists(input_file):
        print(f"  [ERROR] File not found: {input_file}")
        return None

    print(f"  Loading {basename}...")
    try:
        archive = pymapdl_reader.Archive(input_file)
    except Exception as e:
        print(f"  [ERROR] Loading {basename}: {e}")
        return None

    print(f"     Nodes: {archive.n_node}, Elements: {archive.n_elem}")

    # Create mapping from True Node ID to array index
    # This prevents Out of Bounds Crash (Access Violation) when Node IDs are skipped!
    node_mapping = {int(nid): i for i, nid in enumerate(archive.nnum)}

    # Build volumetric grid safely
    cells_formatted = []
    skipped = 0
    for elemdata in archive.elem:
        try:
            # Get the exact index mapping for each node ID (the last 4 items in elem data)
            nodes = [node_mapping[int(n)] for n in elemdata[-4:]]
            cells_formatted.append([4] + nodes)
        except KeyError:
            skipped += 1

    if skipped > 0:
        print(f"     Warning: Skipped {skipped} elements with unresolvable node IDs")

    if not cells_formatted:
        print(f"  [ERROR] No valid elements in {basename}")
        return None

    cells_flat = np.array(cells_formatted, dtype=np.int64).ravel()
    cell_types = np.full(len(cells_formatted), pv.CellType.TETRA.value, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells_flat, cell_types, archive.nodes[:, :3])

    # Clean degenerate geometry
    grid = grid.clean()
    print(f"     Clean grid: {grid.n_points} pts, {grid.n_cells} cells")

    # Extract outer boundary surface (drops all internal tetrahedra)
    surface = grid.extract_surface()
    surface = surface.triangulate()
    print(f"     Surface: {surface.n_points} pts, {surface.n_cells} faces")

    # Save optimized surface
    out_name = basename.replace('.cdb', '_SURFACE_OPTIMIZED.vtp')
    out_path = os.path.join(output_dir, out_name)
    surface.save(out_path)
    print(f"     [OK] Saved: {out_name}")

    return surface


def batch_process(cdb_dir, output_dir):
    """Process all generated_*.cdb files in a directory."""
    pattern = os.path.join(cdb_dir, 'generated_*.cdb')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[ERROR] No generated_*.cdb files found in: {cdb_dir}")
        print(f"   Searched pattern: {pattern}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(files)} generated CDB files")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    results = []
    success = 0
    failed = 0

    for i, filepath in enumerate(files):
        print(f"\n[{i+1}/{len(files)}]")
        surface = process_mesh(filepath, output_dir)
        if surface is not None:
            results.append({
                'file': os.path.basename(filepath),
                'surface': surface,
                'points': surface.n_points,
                'faces': surface.n_cells,
            })
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"SUMMARY")
    print(f"   Processed: {success}")
    print(f"   Failed:    {failed}")
    print(f"   Output:    {output_dir}")

    if results:
        pts = [r['points'] for r in results]
        faces = [r['faces'] for r in results]
        print(f"   Points: min={min(pts)}, mean={np.mean(pts):.0f}, max={max(pts)}")
        print(f"   Faces:  min={min(faces)}, mean={np.mean(faces):.0f}, max={max(faces)}")

    return results


if __name__ == "__main__":
    # Determine paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # thesis_output/ is the parent of output_analysis/
    cdb_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(script_dir, 'optimized_cdb')

    interactive = '--interactive' in sys.argv

    if len(sys.argv) > 1 and sys.argv[1] != '--interactive':
        # Single file mode
        target_file = sys.argv[1]
        if not os.path.isabs(target_file):
            target_file = os.path.join(cdb_dir, target_file)
        os.makedirs(output_dir, exist_ok=True)
        surf = process_mesh(target_file, output_dir)
        if surf and interactive:
            print("\nLaunching 3D viewer...")
            surf.plot(show_edges=True, color='lightblue', background='white')
    else:
        # Batch mode - process all generated CDBs
        results = batch_process(cdb_dir, output_dir)

        if interactive and results:
            # Show the last processed surface in 3D viewer
            last = results[-1]
            print(f"\nLaunching 3D viewer for: {last['file']}")
            last['surface'].plot(show_edges=True, color='lightblue', background='white')
