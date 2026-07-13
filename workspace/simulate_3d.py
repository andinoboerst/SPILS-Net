import argparse
import logging
from fem_sim.tct_tractions_3d import TCTExtractTractions3D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Simulate3D")

def main():
    parser = argparse.ArgumentParser(description="Run full 3D TCT Simulation")
    parser.add_argument("--freq", type=int, default=1000, help="Simulation frequency")
    parser.add_argument("--law", type=str, choices=["elastic", "plastic"], default="elastic", help="Constitutive law")
    args = parser.parse_args()

    logger.info(f"Setting up 3D Simulation with {args.law} material and {args.freq}Hz frequency...")
    
    sim = TCTExtractTractions3D(
        frequency=args.freq,
        constitutive_model=args.law,
        configuration="scaled"
    )

    # sim.time_total = 0.002
    
    logger.info("Running full simulation...")
    sim.run()
    
    logger.info("Generating GIF animation...")
    sim.postprocess("u", "u", "norm", name="simulate_3d_result")

    logger.info("Done! Output saved to simulate_3d_result.gif")

if __name__ == "__main__":
    main()
