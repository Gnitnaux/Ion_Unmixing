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
        "--stage1",
        action="store_true",
        default=False,
        help="Run Stage 1 BYOL pre-training"
    )

    parser.add_argument(
        "--stage2",
        action="store_true",
        default=False,
        help="Run Stage 2 classification fine-tuning"
    )

    parser.add_argument(
        "--re-training",
        action="store_true",
        default=False,
        help="Retrain from scratch (ignore checkpoints)"
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
        print("Stage 2: 5-level concentration classification.")
        data_path = os.path.join('data', args.data)
        # Run both stages if neither --stage1 nor --stage2 specified
        run_stage1 = args.stage1 or (not args.stage1 and not args.stage2)
        run_stage2 = args.stage2 or (not args.stage1 and not args.stage2)
        mix_filter = {'mix_only': False, 'present_conc_range': None}
        run_byol_pipeline(
            data_path, args.model_dir, plot=True,
            **mix_filter,
            stage1=run_stage1, stage2=run_stage2,
            re_training=args.re_training,
            cut_range=(0, 2500),
        )
        print("BYOL pipeline completed.")


    print("\nDone!")


if __name__ == "__main__":
    main()
