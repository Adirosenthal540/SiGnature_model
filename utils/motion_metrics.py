"""
Motion evaluation metrics and scoring functions.

This module contains utility functions for evaluating motion quality,
including motion edit scores, similarity metrics, and other evaluation tools.
"""

import numpy as np
import torch


def calculate_motion_edit_score(
    sample_seg_tar,
    sample_base_model_tar,
    sample_res,
    edited_joints_mask,
    motion_name=None,
    save_detailed_analysis=False,
    save_path=None,
):
    """
    Calculate a score for motion editing quality based on joint-specific distances.

    The score measures:
    1. How close the edited motion (sample_res) is to the target motion (sample_seg_tar)
       at the joints that were supposed to be edited (joints_take_part_in_motion_indexs)
    2. How close the edited motion (sample_res) remains to the base motion (sample_base_model_tar)
       at the joints that were NOT supposed to be edited (other joints)

    Args:
        sample_seg_tar: Target motion tensor [batch, joints, features, frames] or [joints, features, frames]
        sample_base_model_tar: Base motion tensor [batch, joints, features, frames] or [joints, features, frames]
        sample_res: Edited motion tensor [batch, joints, features, frames] or [joints, features, frames]
        edited_joints_mask: Boolean mask indicating which joints were supposed to be edited
        motion_name: Optional name for logging and file naming
        save_detailed_analysis: Whether to save detailed analysis to file
        save_path: Optional path to save detailed analysis

    Returns:
        dict: Score analysis containing:
            - total_score: Overall motion edit quality score (higher is better)
            - target_similarity_score: How well edited motion matches target at edited joints
            - base_preservation_score: How well edited motion preserves base motion at other joints
            - joint_scores: Per-joint analysis
            - detailed_metrics: Additional metrics for analysis
    """
    
    def log_print(*args, **kwargs):
        """Print to console"""
        message = " ".join(str(arg) for arg in args)
        print(message)

    log_print(f"Calculating motion edit score for motion: {motion_name}")

    # Process input tensors
    target_motion = sample_seg_tar.detach().cpu().numpy().squeeze(1)
    base_motion = sample_base_model_tar.detach().cpu().numpy().squeeze(1)
    edited_motion = sample_res.detach().cpu().numpy().squeeze(1)
    edited_joints_mask = edited_joints_mask.detach().cpu().numpy()

    # Ensure all motions have the same shape
    if not (target_motion.shape == base_motion.shape == edited_motion.shape):
        raise ValueError(f"Motion shapes don't match: target={target_motion.shape}, " f"base={base_motion.shape}, edited={edited_motion.shape}")

    # Create mask for joints that should remain unchanged
    preserved_joints_mask = ~edited_joints_mask

    log_print(f"Edited joints: {edited_joints_mask.sum()} joints")
    log_print(f"Preserved joints: {preserved_joints_mask.sum()} joints")

    # Calculate joint-specific scores
    joint_scores = {}
    detailed_metrics = {}
    
    # Target similarity at edited joints
    target_similarity_score = np.linalg.norm((edited_motion - target_motion)[edited_joints_mask])
    
    # Base preservation at preserved joints
    base_preservation_score = np.linalg.norm((edited_motion - base_motion)[preserved_joints_mask])
    
    # Per-joint analysis
    for joint_idx in range(edited_motion.shape[0]):
        joint_scores[joint_idx] = {
            "target_similarity": float(np.linalg.norm(edited_motion[joint_idx] - target_motion[joint_idx])),
            "base_preservation": float(np.linalg.norm(edited_motion[joint_idx] - base_motion[joint_idx])),
            "was_edited": bool(edited_joints_mask[joint_idx])
        }
    
    # Overall scores
    target_weight = 0.7
    base_weight = 0.3
    total_score = target_weight * target_similarity_score + base_weight * base_preservation_score
    
    # Additional detailed metrics
    detailed_metrics = {
        "motion_shape": list(edited_motion.shape),
        "num_edited_joints": int(edited_joints_mask.sum()),
        "num_preserved_joints": int(preserved_joints_mask.sum()),
        "target_weight": target_weight,
        "base_weight": base_weight,
        "motion_name": motion_name
    }
    
    score = {
        "total_score": float(total_score),
        "target_similarity_score": float(target_similarity_score),
        "base_preservation_score": float(base_preservation_score),
        "joint_scores": joint_scores,
        "detailed_metrics": detailed_metrics,
        "motion_name": motion_name,
        "motion_shape": edited_motion.shape
    }
    
    # Save detailed analysis if requested
    if save_detailed_analysis and save_path:
        _save_motion_edit_analysis(score, save_path, motion_name)
    
    return score


def _save_motion_edit_analysis(results, save_path, motion_name):
    """Save detailed motion edit analysis to file"""
    import json
    import os

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Prepare data for JSON serialization
    json_results = {
        "total_score": float(results["total_score"]),
        "target_similarity_score": float(results["target_similarity_score"]),
        "base_preservation_score": float(results["base_preservation_score"]),
        "detailed_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in results["detailed_metrics"].items()},
        "motion_name": results["motion_name"],
        "motion_shape": list(results["motion_shape"]),
        "joint_scores": {
            str(k): {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv for kk, vv in v.items()}
            for k, v in results["joint_scores"].items()
        },
    }

    # Save to JSON file
    with open(save_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Detailed motion edit analysis saved to: {save_path}")


def calculate_motion_similarity(motion1, motion2, method="l2"):
    """
    Calculate similarity between two motion sequences.
    
    Args:
        motion1: First motion tensor [joints, features, frames]
        motion2: Second motion tensor [joints, features, frames]
        method: Similarity method ("l2", "cosine", "dtw")
    
    Returns:
        float: Similarity score
    """
    if method == "l2":
        return float(np.linalg.norm(motion1 - motion2))
    elif method == "cosine":
        # Flatten motions and calculate cosine similarity
        m1_flat = motion1.flatten()
        m2_flat = motion2.flatten()
        return float(np.dot(m1_flat, m2_flat) / (np.linalg.norm(m1_flat) * np.linalg.norm(m2_flat)))
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def calculate_joint_velocity(motion):
    """
    Calculate joint velocities from motion sequence.
    
    Args:
        motion: Motion tensor [joints, features, frames]
    
    Returns:
        np.ndarray: Velocity tensor [joints, features, frames-1]
    """
    return motion[:, :, 1:] - motion[:, :, :-1]


def calculate_motion_smoothness(motion):
    """
    Calculate motion smoothness based on acceleration.
    
    Args:
        motion: Motion tensor [joints, features, frames]
    
    Returns:
        float: Smoothness score (lower is smoother)
    """
    velocity = calculate_joint_velocity(motion)
    acceleration = velocity[:, :, 1:] - velocity[:, :, :-1]
    return float(np.linalg.norm(acceleration))
