#!/usr/bin/env python3
"""
Semantic Weighting Module for Training

This module implements importance weighting based on semantic areas using beat velocity
from the alignmenter to raise weights for semantic regions during training.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from data_loaders.beat2.utils.metric import alignment

from scipy.signal import argrelextrema

class SemanticWeighting:
    """
    Semantic weighting module that uses beat velocity from GT motion to create
    importance weights for semantic areas during training.
    """
    
    def __init__(
        self,
        sigma: float = 0.3,
        order: int = 7,
        avg_vel: Optional[np.ndarray] = None,
        upper_body_joints: list = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        device: str = "cuda"
    ):
        """
        Initialize semantic weighting module.
        
        Args:
            sigma: Sigma parameter for alignment calculation
            order: Order parameter for alignment calculation
            avg_vel: Average velocity for normalization
            upper_body_joints: List of upper body joint indices
            device: Device to use for calculations
        """
        self.device = device
        self.upper_body_joints = upper_body_joints
        
        # Initialize alignmenter for beat velocity calculation
        self.alignmenter = alignment(sigma, order, avg_vel, upper_body_joints)
        
        # Weight parameters
        self.base_weight = 1.0
        self.semantic_multiplier = 2.0  # Multiply weights for semantic areas
        self.min_weight = 0.1
        self.max_weight = 5.0
    
    def calculate_beat_velocity(self, joints_body: np.ndarray, align_mask: int = 0, pose_fps: int = 30) -> np.ndarray:
        """
        Calculate beat velocity from joint positions using the alignmenter.
        
        Args:
            joints_body: Joint positions [n_frames, n_joints, 3]
            align_mask: Mask for alignment calculation
            pose_fps: Pose frame rate
            
        Returns:
            Beat velocity array
        """
        n_frames = joints_body.shape[0]
        
        # Calculate beat velocity using alignmenter
        beat_vel, vel = self.alignmenter.load_pose(
            joints_body.reshape(-1, joints_body.shape[1] * joints_body.shape[2]), 
            align_mask, 
            n_frames - align_mask, 
            pose_fps, 
            True,
            return_vel=True
        )
        
        return beat_vel, vel
    
    def create_semantic_weights(
        self, 
        joints_body: np.ndarray, 
        align_mask: int = 0,
        pose_fps: int = 30,
        semantic_threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Create importance weights based on semantic areas using beat velocity.
        
        Args:
            joints_body: Joint positions [n_frames, n_joints, 3]
            align_mask: Mask for alignment calculation
            pose_fps: Pose frame rate
            semantic_threshold: Threshold for identifying semantic areas
            
        Returns:
            Importance weights tensor [n_frames]
        """
        # Calculate beat velocity
        beat_vel, vel = self.calculate_beat_velocity(joints_body, align_mask, pose_fps)
        data_each_file = np.array(joints_body).reshape(-1, joints_body.shape[1] * joints_body.shape[2])
        
        joints = data_each_file.transpose(1, 0)
        dt = 1/pose_fps
        # first steps is forward diff (t+1 - t) / dt
        init_vel = (joints[:, 1:2] - joints[:, :1]) / dt
        # middle steps are second order (t+1 - t-1) / 2dt
        middle_vel = (joints[:, 2:] - joints[:, 0:-2]) / (2 * dt)
        # last step is backward diff (t - t-1) / dt
        final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
        vel = np.concatenate([init_vel, middle_vel, final_vel], 1).transpose(1, 0).reshape(data_each_file.shape[0], -1, 3)
        #vel = data_each_file.reshape(data_each_file.shape[0], -1, 3)[1:] - data_each_file.reshape(data_each_file.shape[0], -1, 3)[:-1]
        vel = np.linalg.norm(vel, axis=2) / self.alignmenter.mmae
        frame_weights = np.sum(vel>self.alignmenter.threshold, axis=1) / vel.shape[1] * self.semantic_multiplier

        # n_frames = joints_body.shape[0]
        
        # # # Focus on upper body joints for semantic weighting
        # # upper_body_beat_vel = []
        # # for joint_idx in self.upper_body_joints:
        # #     if joint_idx < len(beat_vel):
        # #         upper_body_beat_vel.append(beat_vel[joint_idx])
        
        # if beat_vel:
        #     # Calculate average beat velocity across upper body joints
        #     avg_beat_vel = np.mean(beat_vel, axis=0)
            
        #     # Normalize beat velocity
        #     if np.max(avg_beat_vel) > 0:
        #         normalized_beat_vel = avg_beat_vel / np.max(avg_beat_vel)
        #     else:
        #         normalized_beat_vel = avg_beat_vel
            
        #     # Create weights based on beat velocity
        #     # Higher beat velocity indicates more semantic importance
        #     frame_weights =   + (normalized_beat_vel * self.semantic_multiplier)
            
        #     # Apply threshold for semantic areas
        #     semantic_mask = normalized_beat_vel > semantic_threshold
        #     frame_weights[semantic_mask] *= self.semantic_multiplier
            
        #     # Clip weights to reasonable range
        #     frame_weights = np.clip(frame_weights, self.min_weight, self.max_weight)
        
        return torch.tensor(frame_weights, dtype=torch.float32, device=self.device)
    
    def apply_semantic_weights_to_loss(
        self, 
        loss: torch.Tensor, 
        joints_body: np.ndarray,
        align_mask: int = 0,
        pose_fps: int = 30,
        semantic_threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Apply semantic weights to loss tensor.
        
        Args:
            loss: Loss tensor [batch_size, n_frames, ...]
            joints_body: Joint positions [n_frames, n_joints, 3]
            align_mask: Mask for alignment calculation
            pose_fps: Pose frame rate
            semantic_threshold: Threshold for identifying semantic areas
            
        Returns:
            Weighted loss tensor
        """
        # Create semantic weights
        weights = self.create_semantic_weights(
            joints_body, align_mask, pose_fps, semantic_threshold
        )
        
        # Ensure weights have the right shape for broadcasting
        if loss.dim() > 1:
            # Reshape weights to match loss dimensions
            weight_shape = [1] * loss.dim()
            weight_shape[1] = weights.shape[0]  # Frame dimension
            weights = weights.view(weight_shape)
        
        # Apply weights to loss
        weighted_loss = loss * weights
        
        return weighted_loss
    
    def update_weight_parameters(
        self,
        base_weight: Optional[float] = None,
        semantic_multiplier: Optional[float] = None,
        min_weight: Optional[float] = None,
        max_weight: Optional[float] = None
    ):
        """
        Update weight parameters.
        
        Args:
            base_weight: Base weight for non-semantic areas
            semantic_multiplier: Multiplier for semantic areas
            min_weight: Minimum weight value
            max_weight: Maximum weight value
        """
        if base_weight is not None:
            self.base_weight = base_weight
        if semantic_multiplier is not None:
            self.semantic_multiplier = semantic_multiplier
        if min_weight is not None:
            self.min_weight = min_weight
        if max_weight is not None:
            self.max_weight = max_weight


def create_semantic_weighting_from_dataset(
    dataset,
    sigma: float = 0.3,
    order: int = 7,
    upper_body_joints: list = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    device: str = "cuda"
) -> SemanticWeighting:
    """
    Create semantic weighting module from dataset.
    
    Args:
        dataset: Dataset object with avg_vel attribute
        sigma: Sigma parameter for alignment calculation
        order: Order parameter for alignment calculation
        upper_body_joints: List of upper body joint indices
        device: Device to use for calculations
        
    Returns:
        Initialized SemanticWeighting object
    """
    avg_vel = getattr(dataset, 'avg_vel', None)
    
    return SemanticWeighting(
        sigma=sigma,
        order=order,
        avg_vel=avg_vel,
        upper_body_joints=upper_body_joints,
        device=device
    )


def apply_semantic_weighting_to_training_loss(
    loss_terms: dict,
    joints_body: np.ndarray,
    semantic_weighting: SemanticWeighting,
    align_mask: int = 0,
    pose_fps: int = 30,
    semantic_threshold: float = 0.5,
    weight_keys: list = ["rot_mse", "vel_mse", "rcxyz_mse"]
) -> dict:
    """
    Apply semantic weighting to training loss terms.
    
    Args:
        loss_terms: Dictionary of loss terms
        joints_body: Joint positions [n_frames, n_joints, 3]
        semantic_weighting: SemanticWeighting object
        align_mask: Mask for alignment calculation
        pose_fps: Pose frame rate
        semantic_threshold: Threshold for identifying semantic areas
        weight_keys: List of loss keys to apply weighting to
        
    Returns:
        Dictionary with weighted loss terms
    """
    weighted_loss_terms = loss_terms.copy()
    
    # Apply semantic weighting to specified loss terms
    for key in weight_keys:
        if key in loss_terms:
            weighted_loss_terms[key] = semantic_weighting.apply_semantic_weights_to_loss(
                loss_terms[key],
                joints_body,
                align_mask,
                pose_fps,
                semantic_threshold
            )
    
    return weighted_loss_terms

