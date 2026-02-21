import pyvista as pv
from ansys.mapdl import reader as pymapdl_reader
import os
import sys
import numpy as np

def process_mesh(input_file):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return None

    print(f"Loading {input_file}...")
    try:
        archive = pymapdl_reader.Archive(input_file)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return None

    print(f"Archive loaded! Nodes: {archive.n_node}, Elements: {archive.n_elem}")

    # Create mapping from True Node ID to array index
    # This prevents Out of Bounds Crash (Access Violation) when Node IDs are skipped!
    print("Mapping node topologies...")
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
        print(f"Warning: Skipped {skipped} elements with unresolvable node IDs.")

    cells_flat = np.array(cells_formatted, dtype=np.int64).ravel()
    cell_types = np.full(len(cells_formatted), pv.CellType.TETRA.value, dtype=np.uint8)

    print("Building PyVista Volume Grid...")
    grid = pv.UnstructuredGrid(cells_flat, cell_types, archive.nodes[:, :3])
    
    print("Cleaning node structures to remove degenerate geometry...")
    grid = grid.clean()
    print(f"Clean volume grid points: {grid.n_points}, Cells: {grid.n_cells}")

    print("Extracting outer boundary surface to drastically minimize memory...")
    # (Deprecated warning handled internally)
    
    # Extract outer surface (drops all internal tetrahedra)
    surface = grid.extract_surface()
    
    # Optional decimation - uncomment if files are still too large
    # surface = surface.decimate(0.5) 

    print("Triangulating surface...")
    surface = surface.triangulate()
    
    print(f"Surface extracted! Points: {surface.n_points}, Faces: {surface.n_cells}")

    base_name = os.path.basename(input_file)
    output_vtp = base_name.replace('.cdb', '_SURFACE_OPTIMIZED.vtp')
    surface.save(output_vtp)
    print(f"Success! Saved optimized mesh to: {os.path.abspath(output_vtp)}")

    return surface

if __name__ == "__main__":
    target_file = r'..\4_bonemat_cdb_files\AB029_left_bonemat.cdb'
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        
    surf = process_mesh(target_file)
    if surf:
         # Launch interactive viewer
         print("\nLaunching 3D window...\nClose the pop-up window to exit the script.")
         surf.plot(show_edges=True, color='lightblue', background='white')
