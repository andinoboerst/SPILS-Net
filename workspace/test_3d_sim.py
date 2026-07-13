import numpy as np
import sys
import os

# Add workspace to path
sys.path.append("/workspace/workspace")

from fem_sim.tct_sims_3d import TCTSimulation3D

def main():
    sim = TCTSimulation3D(frequency=1000, constitutive_model="elastic", configuration="scaled")
    sim.num_steps = 10
    sim.dt = 1e-4
    print("Initializing 3D simulation...")
    print("Running some steps...")
    sim.run()
    
    print("Mesh dim:", sim.dim)
    print("Number of cells:", sim.mesh.topology.index_map(sim.dim).size_local)
    print("Number of interface nodes:", len(sim.interface_nodes))
    
    print("Success!")

if __name__ == "__main__":
    main()
