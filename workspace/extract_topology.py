import sys
import numpy as np
from pathlib import Path
import logging

from fem_sim.tct_tractions_3d import TCTExtractTractions3D
from create_predictor_3d import Sim3DConfig

DATA_FOLDER = Path("training_data_3d")
DATA_FOLDER.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Extractor")

def main():
    cfg = Sim3DConfig(constitutive_law="elastic", frequency=500, time_total=0.0005)
    from create_predictor_3d import _make_extractor
    tct = _make_extractor(cfg)
    
    # Just initialize the geometry
    logger.info("Initializing mesh and boundaries...")
    tct._define_mesh()
    tct._define_functionspace()
    tct._preprocess()
    
    logger.info("Extracting topology...")
    coords, connectivity = tct.get_interface_topology()
    
    out_file = DATA_FOLDER / f"interface_topology_{cfg.constitutive_law}.npz"
    np.savez_compressed(out_file, coords=coords, connectivity=connectivity)
    logger.info(f"Saved to {out_file}. Coordinates shape: {coords.shape}, Connectivity shape: {connectivity.shape}")

if __name__ == "__main__":
    main()
