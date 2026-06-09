"""
3D Visualization module for training loss curves and loss landscapes.

This module provides utilities for creating 3D plots including:
- 3D loss surface plots showing how loss varies with epoch and learning rate
- 3D loss trajectory comparisons across multiple models
- Interactive 3D scatter and surface visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple
import os


class Loss3DVisualizer:
    """Handles 3D visualization of training loss curves."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 9)):
        """Initialize the 3D visualizer."""
        self.figsize = figsize
        self.color_map = plt.cm.Set3
    
    def plot_3d_loss_surface(self, 
                            losses: List[float],
                            learning_rates: List[float],
                            model_name: str,
                            save_path: Optional[str] = None) -> plt.Figure:
        """Create a 3D surface plot of loss landscape."""
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        epochs = np.arange(len(losses))
        
        if len(learning_rates) == 1:
            lr_range = np.linspace(learning_rates[0] * 0.5, learning_rates[0] * 1.5, 5)
        else:
            lr_range = np.array(learning_rates)
        
        epochs_mesh, lr_mesh = np.meshgrid(epochs, lr_range)
        
        loss_surface = np.zeros_like(epochs_mesh, dtype=float)
        for i, lr in enumerate(lr_range):
            lr_factor = 1.0 + (lr - learning_rates[0]) * 100
            loss_surface[i, :] = np.array(losses) * np.clip(lr_factor, 0.5, 2.0)
        
        surf = ax.plot_surface(epochs_mesh, lr_mesh, loss_surface,
                              cmap='viridis', alpha=0.8, edgecolor='none')
        
        fig.colorbar(surf, ax=ax, label='Loss', shrink=0.5)
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
        ax.set_zlabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title(f'3D Loss Surface - {model_name.upper()}', 
                    fontsize=13, fontweight='bold', pad=20)
        
        ax.view_init(elev=25, azim=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D loss surface saved to {save_path}")
        
        return fig
    
    def plot_3d_loss_trajectory(self,
                               training_losses_dict: Dict[str, List[float]],
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create a 3D plot comparing loss trajectories of multiple models."""
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        model_names = list(training_losses_dict.keys())
        colors = self.color_map(np.linspace(0, 1, len(model_names)))
        
        for i, (model_name, losses) in enumerate(training_losses_dict.items()):
            losses = np.array(losses)
            epochs = np.arange(len(losses))
            model_indices = np.full_like(epochs, i, dtype=float)
            
            ax.plot(epochs, model_indices, losses,
                   color=colors[i], marker='o', markersize=4,
                   linewidth=2.5, label=model_name, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Model Index', fontsize=11, fontweight='bold')
        ax.set_zlabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title('3D Loss Trajectory Comparison - All Models',
                    fontsize=13, fontweight='bold', pad=20)
        
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontsize=9)
        
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)
        
        ax.view_init(elev=20, azim=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D loss trajectory saved to {save_path}")
        
        return fig
    
    def plot_3d_loss_scatter_multi(self,
                                  training_losses_dict: Dict[str, List[float]],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Create an advanced 3D scatter plot with model loss trajectories."""
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        model_names = list(training_losses_dict.keys())
        colors = self.color_map(np.linspace(0, 1, len(model_names)))
        
        for i, (model_name, losses) in enumerate(training_losses_dict.items()):
            losses = np.array(losses)
            epochs = np.arange(len(losses))
            model_indices = np.full_like(epochs, i, dtype=float)
            
            loss_min, loss_max = losses.min(), losses.max()
            normalized_losses = (losses - loss_min) / (loss_max - loss_min + 1e-7)
            sizes = 20 + normalized_losses * 100
            
            ax.scatter(epochs, model_indices, losses,
                      c=[colors[i]] * len(epochs),
                      s=sizes, alpha=0.6, edgecolors='black',
                      linewidth=0.5, label=model_name)
            
            ax.plot(epochs, model_indices, losses,
                   color=colors[i], alpha=0.3, linewidth=1.5, linestyle='--')
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Model Index', fontsize=11, fontweight='bold')
        ax.set_zlabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title('3D Loss Analysis - Multi-Model Comparison',
                    fontsize=13, fontweight='bold', pad=20)
        
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontsize=9)
        
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)
        ax.view_init(elev=25, azim=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D scatter plot saved to {save_path}")
        
        return fig
    
    def plot_3d_loss_convergence(self,
                                training_losses_dict: Dict[str, List[float]],
                                save_path: Optional[str] = None) -> plt.Figure:
        """Create a specialized 3D plot focusing on convergence patterns."""
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        model_names = list(training_losses_dict.keys())
        
        for i, (model_name, losses) in enumerate(training_losses_dict.items()):
            losses = np.array(losses)
            epochs = np.arange(len(losses))
            model_indices = np.full_like(epochs, i, dtype=float)
            
            epoch_colors = plt.cm.RdYlGn_r(epochs / len(epochs))
            
            for j in range(len(epochs) - 1):
                ax.plot(epochs[j:j+2], model_indices[j:j+2], losses[j:j+2],
                       color=epoch_colors[j], linewidth=3, marker='o',
                       markersize=6, alpha=0.8)
            
            ax.scatter([epochs[-1]], [model_indices[-1]], [losses[-1]],
                      c='red', s=100, marker='*', edgecolors='darkred',
                      linewidth=1, label=f'{model_name} (final)', zorder=5)
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Model', fontsize=11, fontweight='bold')
        ax.set_zlabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title('3D Convergence Pattern Analysis',
                    fontsize=13, fontweight='bold', pad=20)
        
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontsize=9)
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                                   norm=plt.Normalize(vmin=0, vmax=len(list(training_losses_dict.values())[0])))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label='Epoch Progress', shrink=0.6)
        
        ax.view_init(elev=25, azim=135)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D convergence pattern saved to {save_path}")
        
        return fig
    
    @staticmethod
    def create_3d_visualization_directory(base_dir: str) -> str:
        """Create a directory for 3D visualizations if it doesn't exist."""
        viz_dir = os.path.join(base_dir, '3d_visualizations')
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        return viz_dir
