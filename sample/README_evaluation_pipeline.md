# Evaluation Pipeline for Motion Transfer

This pipeline processes all `insert_sg_info_with_a_pose` data and runs motion transfer, saving NPZ files for evaluation and videos for debugging.

## Overview

The pipeline:
1. Loads the dataset and model
2. Processes each person's data to find semantic gestures
3. For each gesture, runs motion transfer using the base model
4. Saves NPZ files to an `evaluation` folder for later evaluation
5. Saves videos to a `debug` folder for visual inspection
6. Names all files with the person's name for easy identification

## Files

- `evaluation_pipeline.py`: Main pipeline implementation
- `run_evaluation.py`: Easy-to-use script for running the pipeline
- `README.md`: This documentation

## Usage

### Basic Usage

```bash
python sample/run_evaluation.py --model_path /path/to/your/model.pt
```

### Advanced Usage

```bash
python sample/run_evaluation.py \
    --model_path /path/to/your/model.pt \
    --dataset beat2 \
    --test_data_name "person_name" \
    --output_base_path ./my_evaluation_results \
    --device 0 \
    --seed 42
```

### Parameters

- `--model_path`: Path to the model file (required)
- `--dataset`: Dataset name (default: "beat2")
- `--config_seg_opt_path`: Path to segmentation config (default: "./datasets/SeG_SMPLX/config_seg_opt.yaml")
- `--test_data_name`: Process only specific test data (optional)
- `--output_base_path`: Base path for output files (default: "./evaluation_results")
- `--device`: CUDA device ID (default: 0)
- `--seed`: Random seed (default: 42)

## Output Structure

```
evaluation_results/
└── evaluation_YYYYMMDD_HHMMSS/
    ├── evaluation/          # NPZ files for evaluation
    │   ├── person1_gesture1_0_90_motion_transfer.npz
    │   ├── person1_gesture1_0_90_target.npz
    │   ├── person1_gesture2_30_120_motion_transfer.npz
    │   └── ...
    └── debug/               # Videos for debugging
        ├── person1_gesture1_0_90.mp4
        ├── person1_gesture2_30_120.mp4
        └── ...
```

## File Naming Convention

- **NPZ files**: `{person_name}_{gesture_name}_{start_idx}_{end_idx}_{type}.npz`
  - `motion_transfer.npz`: Generated motion transfer result
  - `target.npz`: Original target motion for comparison
- **Video files**: `{person_name}_{gesture_name}_{start_idx}_{end_idx}.mp4`

## Key Differences from Original Code

1. **Batch Processing**: Processes all gestures for all persons automatically
2. **Organized Output**: Saves files in structured directories
3. **Person-based Naming**: All files include the person's name
4. **Memory Management**: Clears model cache between gestures to prevent OOM

## Requirements

- Same requirements as the main SiGnature project
- CUDA-compatible GPU recommended
- Sufficient disk space for NPZ and video files

## Troubleshooting

### Memory Issues
If you encounter out-of-memory errors, the pipeline automatically clears the model cache between gestures. If issues persist, try:
- Reducing batch size
- Processing fewer gestures at once
- Using a GPU with more memory

### Missing Files
Ensure all required dataset files are present:
- Segmentation dataset files
- Model checkpoint
- Configuration files

### Video Generation Issues
If video generation fails, the pipeline will continue processing other gestures. Check:
- SMPLX model files are available
- Rendering dependencies are installed
- Sufficient disk space for video files

## Example Output

```
🚀 Starting Evaluation Pipeline...
📁 Created evaluation directory: ./evaluation_results/evaluation_20250104_120000/evaluation
📁 Created debug directory: ./evaluation_results/evaluation_20250104_120000/debug
📦 Loading dataset...
🧠 Loading model...

👤 Processing person: john_doe
✅ Processed gesture: handshake_01 for person: john_doe
📁 Saved NPZ: ./evaluation_results/evaluation_20250104_120000/evaluation/john_doe_handshake_01_0_90_motion_transfer.npz
✅ Generated video: ./evaluation_results/evaluation_20250104_120000/debug/john_doe_handshake_01_0_90.mp4
✅ Processed 3 gestures for john_doe

🎉 Pipeline completed!
📊 Processed 5 persons
📊 Processed 15 gestures total
📁 Evaluation files saved in: ./evaluation_results/evaluation_20250104_120000/evaluation
📁 Debug videos saved in: ./evaluation_results/evaluation_20250104_120000/debug
```
