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
        choices=["read", "plsr_unmixing"],
        default="read",
        help="Operation mode: read or plsr_unmixing"
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
        # Placeholder for PLSR unmixing implementation
        print("PLSR unmixing functionality is not yet implemented.")


    print("\nDone!")


if __name__ == "__main__":
    main()
