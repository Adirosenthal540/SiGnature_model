#!/usr/bin/env python3
"""
Usage script for the evaluation pipeline.

This script provides an easy way to run the evaluation pipeline with different configurations.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.seg_evaluation_pipeline import run_evaluation_pipeline
from utils.parser_util import generate_args


def main():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline for motion transfer")

    # Model and data arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset", type=str, default="beat2", help="Dataset name")
    parser.add_argument("--config_seg_opt_path", type=str, default="./datasets/SeG_SMPLX/config_seg_opt.yaml", help="Path to segmentation config")

    # Test data filtering
    parser.add_argument("--test_data_name", type=str, default=None, help="Specific test data name to process (optional)")

    # Output arguments
    parser.add_argument("--output_base_path", type=str, default="./evaluation_results", help="Base path for output files")

    # Technical arguments
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create a mock args object with the required attributes
    class Args:
        def __init__(self):
            self.model_path = args.model_path
            self.dataset = args.dataset
            self.config_seg_opt_path = args.config_seg_opt_path
            self.test_data_name = args.test_data_name
            self.device = args.device
            self.seed = args.seed
            self.use_seg = True
            self.batch_size = 1
            self.handshake_size = 30
            self.blend_len = 10
            self.skip_steps_double_take = 100
            self.guidance_param = 1.0
            self.num_repetitions = 1

    eval_args = Args()

    print("🚀 Starting Evaluation Pipeline...")
    print(f"📁 Model path: {args.model_path}")
    print(f"📁 Output base path: {args.output_base_path}")
    print(f"🎯 Test data name: {args.test_data_name or 'All'}")
    print(f"🔧 Device: {args.device}")
    print(f"🎲 Seed: {args.seed}")

    try:
        run_evaluation_pipeline(eval_args, output_base_path=args.output_base_path)
        print("✅ Pipeline completed successfully!")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
