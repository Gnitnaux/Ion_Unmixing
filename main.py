#!/usr/bin/env python3
"""
Main program interface for Transfer Learning Assisted SERS
This script serves as the main entry point for the SERS analysis pipeline.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np

from src.utils import read_data, preprocess_data
from src.plsr_unmixing import run_plsr_unmixing
from src.byol_pipeline import run_byol_pipeline

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Main function to orchestrate the SERS analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="ML for Ion Unmixing - X.T.Liu 20260615"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["read", "plsr_unmixing", "byol_pipeline"],
        default="read",
        help="Operation mode: read, plsr_unmixing, or byol_pipeline"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="data_mix",
        help="Path to preprocessed data directory"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="model",
        help="Path to model directory"
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "quantification"],
        default="classification",
        help="Task for teacher_student mode: classification or quantification"
    )

    parser.add_argument(
        "--unfreeze",
        action="store_true",
        default=False,
        help="Enable Stage 2 Phase 2: unfreeze encoder for end-to-end fine-tuning"
    )

    parser.add_argument(
        "--unfreeze-epochs",
        type=int,
        default=50,
        help="Number of epochs for unfreeze phase (default: 50, only used with --unfreeze)"
    )

    args = parser.parse_args()
    
    print("=" * 60)
    print("ML for Ion Unmixing - X.T.Liu 20260615")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Data directory: {args.data}")
    print(f"Model directory: {args.model_dir}")
    print("=" * 60)
    
    if args.mode == "read":
        print("\nReading mode selected.")
        group_numbers, concentrations, Raman_Shifts, intensities = read_data(
            os.path.join('data', args.data)
        )
        print(f"Groups: {len(set(group_numbers))}, "
              f"Spectra: {len(intensities)}, "
              f"Raman points: {len(Raman_Shifts)}")
        cut_Raman_Shifts, processed_intensities = preprocess_data(
            Raman_Shifts, intensities, cut_range=(0, 2000), plot=True
        )

        
    elif args.mode == "plsr_unmixing":
        print("\nPLSR Unmixing mode selected.")
        print("Multi-output PLSR (concentration + ratio) for Cu/Fe/Zn.")
        data_path = os.path.join('data', args.data)
        mix_filter = {'mix_only': False, 'present_conc_range': None}
        run_plsr_unmixing(
            data_path, args.model_dir, plot=True,
            **mix_filter,
            peak_position=250, peak_range=20,
            cut_range=(0, 2000)
        )
        print("PLSR unmixing completed.")

    elif args.mode == "byol_pipeline":
        print("\nBYOL Peak-Token Pipeline mode selected.")
        print("Stage 1: BYOL pre-training on peak tokens.")
        print("Stage 2: 5-level concentration classification "
              "(frozen encoder, plan B).")
        data_path = os.path.join('data', args.data)
        mix_filter = {'mix_only': False, 'present_conc_range': None}
        # Build config override for unfreeze
        config_override = {}
        if args.unfreeze:
            config_override["stage2_full_epochs"] = args.unfreeze_epochs
            print(f"  Phase 2 unfreeze enabled: {args.unfreeze_epochs} epochs")
        else:
            print("  Phase 2 unfreeze: disabled (use --unfreeze to enable)")
        run_byol_pipeline(
            data_path, args.model_dir, plot=True,
            **mix_filter,
            stage1=True, stage2=True,
            re_training=True,
            config=config_override,
            cut_range=(0, 2500),
        )
        print("BYOL pipeline completed.")


    print("\nDone!")


if __name__ == "__main__":
    main()
