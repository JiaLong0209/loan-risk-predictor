import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import Dict, List, Tuple
from .app import LoanRiskPredictor
from .config.config import ConfigManager
from sklearn.metrics import confusion_matrix
import torch
import seaborn as sns
import os
import pandas as pd

class LoanRiskPredictorGUI:
    def __init__(self):
        self.config = ConfigManager()
        self.predictor = LoanRiskPredictor()
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Loan Risk Predictor")
        self.root.geometry("1600x900")

        
        ctk.set_appearance_mode("Dark")
        # Set matplotlib style
        plt.style.use('ggplot')

 # Create frames
        self.create_control_frame()
        self.create_results_frame()
        self.create_charts_frame()
        
        # Initialize results storage
        self.results: Dict[str, Dict[str, List[float]]] = {}
        self.confusion_matrices: Dict[str, np.ndarray] = {}
        self.training_losses: Dict[str, List[float]] = {}  # Store training losses for each model
        
        # Debug mode flag
        self.debug_mode = False
        
    
    def create_control_frame(self):
        """Create the control frame with buttons and options."""
        control_frame = ctk.CTkFrame(self.root)
        control_frame.pack(side=ctk.LEFT, fill=ctk.Y, padx=20, pady=20)
        
        # Add test evaluation button
        # self.evaluate_test_btn = ctk.CTkButton(
        #     control_frame,
        #     text="Evaluate Test Data",
        #     command=self.evaluate_test_data,
        #     width=200
        # )
        # self.evaluate_test_btn.pack(pady=10)
        
        # Debug mode toggle
        self.debug_var = ctk.BooleanVar(value=False)
        debug_checkbox = ctk.CTkCheckBox(
            control_frame,
            text="Debug Mode",
            variable=self.debug_var,
            command=self.toggle_debug_mode
        )
        debug_checkbox.pack(pady=5)
        
        # Feature engineering toggle
        self.feature_eng_var = ctk.BooleanVar(value=True)
        feature_eng_checkbox = ctk.CTkCheckBox(
            control_frame,
            text="Enable Feature Engineering",
            variable=self.feature_eng_var,
        )
        feature_eng_checkbox.pack(pady=5)
        
        # K-fold toggle
        self.use_kfold_var = ctk.BooleanVar(value=True)
        kfold_checkbox = ctk.CTkCheckBox(
            control_frame,
            text="Use K-Fold Cross Validation",
            variable=self.use_kfold_var,
            command=self.toggle_kfold_controls
        )
        kfold_checkbox.pack(pady=5)
        
        # Train split rate
        train_frame = ctk.CTkFrame(control_frame)
        train_frame.pack(pady=5, fill=ctk.X)
        ctk.CTkLabel(train_frame, text="Train Split Rate (%)").pack(side=ctk.LEFT, padx=5)
        self.train_split = ctk.CTkSlider(train_frame, from_=50, to=90, number_of_steps=8)
        self.train_split.set(80)
        self.train_split.pack(side=ctk.LEFT, padx=5, fill=ctk.X, expand=True)
        self.train_split_label = ctk.CTkLabel(train_frame, text="80%")
        self.train_split_label.pack(side=ctk.LEFT, padx=5)
        self.train_split.configure(command=self.update_train_split_label)
        
        # Train split rate entry
        train_entry_frame = ctk.CTkFrame(control_frame)
        train_entry_frame.pack(pady=5, fill=ctk.X)
        ctk.CTkLabel(train_entry_frame, text="Enter Train Split:").pack(side=ctk.LEFT, padx=5)
        self.train_split_entry = ctk.CTkEntry(train_entry_frame, width=60)
        self.train_split_entry.pack(side=ctk.LEFT, padx=5)
        self.train_split_entry.insert(0, "80")
        self.train_split_entry.bind('<Return>', self.update_train_split_from_entry)
        
        # K-fold number frame
        self.kfold_frame = ctk.CTkFrame(control_frame)
        self.kfold_frame.pack(pady=5, fill=ctk.X)
        ctk.CTkLabel(self.kfold_frame, text="K-Fold Number").pack(side=ctk.LEFT, padx=5)
        self.k_fold = ctk.CTkSlider(self.kfold_frame, from_=2, to=10, number_of_steps=8)
        self.k_fold.set(5)
        self.k_fold.pack(side=ctk.LEFT, padx=5, fill=ctk.X, expand=True)
        self.k_fold_label = ctk.CTkLabel(self.kfold_frame, text="5")
        self.k_fold_label.pack(side=ctk.LEFT, padx=5)
        self.k_fold.configure(command=self.update_kfold_label)
        
        # K-fold entry frame
        self.kfold_entry_frame = ctk.CTkFrame(control_frame)
        self.kfold_entry_frame.pack(pady=5, fill=ctk.X)
        ctk.CTkLabel(self.kfold_entry_frame, text="Enter K-Fold:").pack(side=ctk.LEFT, padx=5)
        self.k_fold_entry = ctk.CTkEntry(self.kfold_entry_frame, width=60)
        self.k_fold_entry.pack(side=ctk.LEFT, padx=5)
        self.k_fold_entry.insert(0, "5")
        self.k_fold_entry.bind('<Return>', self.update_kfold_from_entry)
        
        # Subsample rate
        subsample_frame = ctk.CTkFrame(control_frame)
        subsample_frame.pack(pady=5, fill=ctk.X)
        ctk.CTkLabel(subsample_frame, text="Subsample Rate (%)").pack(side=ctk.LEFT, padx=5)
        self.subsample_rate = ctk.CTkSlider(subsample_frame, from_=0.1, to=100, number_of_steps=999)
        self.subsample_rate.set(100)
        self.subsample_rate.pack(side=ctk.LEFT, padx=5, fill=ctk.X, expand=True)
        self.subsample_label = ctk.CTkLabel(subsample_frame, text="100%")
        self.subsample_label.pack(side=ctk.LEFT, padx=5)
        self.subsample_rate.configure(command=self.update_subsample_label)
        
        # Subsample rate entry
        subsample_entry_frame = ctk.CTkFrame(control_frame)
        subsample_entry_frame.pack(pady=5, fill=ctk.X)
        ctk.CTkLabel(subsample_entry_frame, text="Enter Subsample:").pack(side=ctk.LEFT, padx=5)
        self.subsample_entry = ctk.CTkEntry(subsample_entry_frame, width=60)
        self.subsample_entry.pack(side=ctk.LEFT, padx=5)
        self.subsample_entry.insert(0, "100")
        self.subsample_entry.bind('<Return>', self.update_subsample_from_entry)
        
        # Model selection
        model_frame = ctk.CTkFrame(control_frame)
        model_frame.pack(pady=5, fill=ctk.X)
        ctk.CTkLabel(model_frame, text="Select Models").pack(anchor=ctk.W, pady=5)
        self.model_vars = {}

        # model_names = ["d_lstm_mlp", "d_lstm", "mlp", "cnn_lightgbm", "dnn", "rnn", "random_forest", "xgboost"]
        model_names = ["d_lstm_mlp", "d_lstm", "mlp", "cnn_lightgbm", "rnn", "random_forest", "xgboost"]
        
        for model in model_names:
            var = ctk.BooleanVar(value=True)
            self.model_vars[model] = var
            ctk.CTkCheckBox(model_frame, text=model, variable=var).pack(anchor=ctk.W, pady=2)
        

        # Run button
        self.run_button = ctk.CTkButton(
            control_frame,
            text="Run Models",
            command=self.run_models
        )
        self.run_button.pack(pady=20)
    
    def update_train_split_from_entry(self, event=None):
        """Update train split from entry field."""
        try:
            value = float(self.train_split_entry.get())
            if 50 <= value <= 90:
                self.train_split.set(value)
                self.update_train_split_label(value)
        except ValueError:
            pass
    
    def update_kfold_from_entry(self, event=None):
        """Update K-fold from entry field."""
        try:
            value = int(self.k_fold_entry.get())
            if 2 <= value <= 10:
                self.k_fold.set(value)
                self.update_kfold_label(value)
        except ValueError:
            pass

    def update_subsample_from_entry(self, event=None):
        """Update subsample rate from entry field."""
        try:
            value = float(self.subsample_entry.get())
            if 0.1 <= value <= 100:
                self.subsample_rate.set(value)
                self.update_subsample_label(value)
        except ValueError:
            pass

    def update_train_split_label(self, value):
        """Update the train split rate label."""
        self.train_split_label.configure(text=f"{int(value)}%")
        self.train_split_entry.delete(0, ctk.END)
        self.train_split_entry.insert(0, str(int(value)))

    def update_kfold_label(self, value):
        """Update the K-fold number label."""
        self.k_fold_label.configure(text=str(int(value)))
        self.k_fold_entry.delete(0, ctk.END)
        self.k_fold_entry.insert(0, str(int(value)))

    def update_subsample_label(self, value):
        """Update the subsample rate label."""
        rounded_value = round(value, 1)
        self.subsample_label.configure(text=f"{rounded_value:.1f}%")
        self.subsample_entry.delete(0, ctk.END)
        self.subsample_entry.insert(0, f"{rounded_value:.1f}")

    def toggle_debug_mode(self):
        """Toggle debug mode."""
        self.debug_mode = self.debug_var.get()
        if self.debug_mode:
            self.debug_text.configure(state=ctk.NORMAL)
            self.debug_text.delete("1.0", ctk.END)
            self.debug_text.insert(ctk.END, "Debug mode enabled\n")
            self.debug_text.configure(state=ctk.DISABLED)
            # Set the callback for debug logger
            self.predictor.debug_logger.set_gui_callback(self.log_debug_message)
        else:
            self.debug_text.configure(state=ctk.NORMAL)
            self.debug_text.delete("1.0", ctk.END)
            self.debug_text.configure(state=ctk.DISABLED)
            # Remove the callback
            self.predictor.debug_logger.set_gui_callback(None)

    def log_debug_message(self, message):
        """Log a debug message to the debug text box."""
        if self.debug_mode:
            self.debug_text.configure(state=ctk.NORMAL)
            self.debug_text.insert(ctk.END, message + "\n")
            self.debug_text.see(ctk.END)  # Scroll to the end
            self.debug_text.configure(state=ctk.DISABLED)

    def create_results_frame(self):
        """Create the results display frame."""
        results_frame = ctk.CTkFrame(self.root)
        results_frame.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=0, pady=20)

        # Results text
        ctk.CTkLabel(results_frame, text="Results").pack(pady=5)
        self.results_text = ctk.CTkTextbox(results_frame, width=450, height=400)
        self.results_text.pack(padx=0, pady=0)

        # Debug text
        ctk.CTkLabel(results_frame, text="Debug Messages").pack(pady=5)
        self.debug_text = ctk.CTkTextbox(results_frame, width=450, height=400)
        self.debug_text.pack(padx=0, pady=0)
        self.debug_text.configure(state=ctk.DISABLED)
    
    def create_charts_frame(self):
        """Create the charts display frame."""
        charts_frame = ctk.CTkFrame(self.root)
        charts_frame.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True, padx=20, pady=20)
        
        # Create tabview for different chart types
        self.tabview = ctk.CTkTabview(charts_frame)
        self.tabview.pack(fill=ctk.BOTH, expand=True)
        
        # Create tabs
        self.tabview.add("Performance Metrics")
        self.tabview.add("Model Comparison")
        self.tabview.add("Training Losses")
        self.tabview.add("Training Time")  # Add new tab for training time

        figsize1 = (15, 20)
        figsize2 = (10, 10)
        figsize3 = (12, 8)  # Size for training loss plot
        figsize4 = (10, 6)  # Size for training time plot
        
        # Create figures for each tab
        self.fig1 = plt.figure(figsize=figsize1)
        self.ax1 = self.fig1.add_subplot(211)  # Regular subplot for bar chart
        self.ax2 = self.fig1.add_subplot(212)  # Horizontal bar chart
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.tabview.tab("Performance Metrics"))
        self.canvas1.get_tk_widget().pack(fill=ctk.BOTH, expand=True)
        
        # Create figure for radar chart
        self.fig2 = plt.figure(figsize=figsize2)
        self.ax3 = self.fig2.add_subplot(111, projection='polar')  # Radar chart
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.tabview.tab("Model Comparison"))
        self.canvas2.get_tk_widget().pack(fill=ctk.BOTH, expand=True)

        # Create figure for training losses
        self.fig3 = plt.figure(figsize=figsize3)
        self.ax4 = self.fig3.add_subplot(111)  # Training loss plot
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.tabview.tab("Training Losses"))
        self.canvas3.get_tk_widget().pack(fill=ctk.BOTH, expand=True)

        # Create figure for training time
        self.fig4 = plt.figure(figsize=figsize4)
        self.ax5 = self.fig4.add_subplot(111)  # Training time plot
        self.canvas4 = FigureCanvasTkAgg(self.fig4, master=self.tabview.tab("Training Time"))
        self.canvas4.get_tk_widget().pack(fill=ctk.BOTH, expand=True)

        # Adjust layout to accommodate the legend
        self.fig1.subplots_adjust(left=0.1, right=0.8, hspace=0.2, wspace=0.2, top=0.95, bottom=0.05)
        self.fig2.subplots_adjust(right=0.9)
        self.fig3.subplots_adjust(right=0.95)  # Adjust for legend
        self.fig4.subplots_adjust(right=0.95)  # Adjust for legend

    def update_charts(self):
        """Update the charts with the latest results."""
        # Clear previous charts
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()  # Clear training time plot
        
        # Prepare data for charts
        models = list(self.results.keys())
        avg_metrics = {
            'accuracy': [np.mean(self.results[m]['accuracy']) for m in models],
            'recall': [np.mean(self.results[m]['recall']) for m in models],
            'precision': [np.mean(self.results[m]['precision']) for m in models],
            'f1_score': [np.mean(self.results[m]['f1_score']) for m in models]
        }

        # Prepare training time data
        training_times = []
        for model in models:
            if 'timing' in self.results[model]:
                timing = self.results[model]['timing']
                if 'total_time' in timing:
                    training_times.append(timing['total_time'])
                else:
                    training_times.append(0)
            else:
                training_times.append(0)

        model_colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        fontsize = 8
        rotation = 23
        facecolors = ['#f2f2f2', '#fafafa']
        metrics_colors = ['#a7ed8a', '#e6eb88' ,'#88ebeb' ,'#f09595']
        grid_colors = ["#a0a0a0", "white"]
        
        # First tab: Performance Metrics
        # Bar chart for average metrics (ax1)
        x = np.arange(len(models))
        width = 0.15  # Adjusted width for 4 metrics
        bars1 = self.ax1.bar(x - 1.5*width, avg_metrics['accuracy'], width, label='Accuracy', color=metrics_colors[0])
        bars2 = self.ax1.bar(x - 0.5*width, avg_metrics['recall'], width, label='Recall', color=metrics_colors[1])
        bars3 = self.ax1.bar(x + 0.5*width, avg_metrics['precision'], width, label='Precision', color=metrics_colors[2])
        bars4 = self.ax1.bar(x + 1.5*width, avg_metrics['f1_score'], width, label='F1-Score', color=metrics_colors[3])
        
        # Add value labels on top of bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                self.ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom', fontsize=fontsize -2, rotation=0 , color='#222222')
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        add_value_labels(bars4)
        
        self.ax1.set_ylabel('Score', fontsize=fontsize)
        self.ax1.set_title('Average Model Performance Metrics', fontsize=fontsize*1.5)
        self.ax1.set_facecolor(facecolors[0])
        self.ax1.set_xticks(x)
        self.ax1.set_xticklabels(models, rotation=rotation, fontsize=fontsize)
        self.ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 0.4), facecolor=facecolors[1], fontsize=fontsize)
        self.ax1.grid(True, linestyle='--', alpha=0.7, color=grid_colors[1])

        # Horizontal bar chart (ax2)
        y = np.arange(4)  
        height = 0.65 / len(models)  # Adjust height based on number of models

        for i, model in enumerate(models):
            metrics = [avg_metrics['accuracy'][i], avg_metrics['recall'][i], avg_metrics['precision'][i], avg_metrics['f1_score'][i]]
            bars = self.ax2.barh(y + i * height, metrics, height, label=model, color=model_colors[i])

            # Add value labels
            for bar in bars:
                width = bar.get_width()
                self.ax2.text(width, bar.get_y() + bar.get_height()/2.,
                            f'{width:.3f}',
                            ha='left', va='center', fontsize=fontsize-1, color='#222222')
        
        self.ax2.set_facecolor(facecolors[1])
        self.ax2.set_xlabel('Score', fontsize=fontsize)
        self.ax2.set_title('Model Performance Comparison', fontsize=fontsize*1.3)
        self.ax2.set_yticks(y + height * (len(models) - 1) / 2)
        self.ax2.set_yticklabels(['Accuracy', 'Recall', 'Precision', 'F1-Score'], fontsize=fontsize, rotation=rotation)

        # Reversing Legend Order
        handles, labels = self.ax2.get_legend_handles_labels()
        self.ax2.legend(handles[::-1], labels[::-1],
                        loc='upper left',
                        bbox_to_anchor=(1.0, 0.4),
                        facecolor=facecolors[1],
                        fontsize=fontsize)
        self.ax2.grid(True, linestyle='--', alpha=0.7, color=grid_colors[1])

        # Radar chart (ax3)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        num_metrics = len(metrics)
        
        # Compute angle for each metric
        angles = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
        angles += angles[:1]  # Close the loop
        
        # Set up the radar chart
        self.ax3.set_theta_offset(np.pi / 2)  # Start from top
        self.ax3.set_theta_direction(-1)  # Clockwise
        
        # Draw axis lines for each metric
        self.ax3.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=fontsize, rotation=45)
        
        # Draw y-axis labels (0-1)
        self.ax3.set_ylim(0, 1)
        self.ax3.set_facecolor(facecolors[0])
        self.ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        self.ax3.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=fontsize)
        self.ax3.grid(True, linestyle='--', alpha=0.7, color=grid_colors[0])
        
        # Plot each model's metrics
        for i, model in enumerate(models):
            values = [
                avg_metrics['accuracy'][i],
                avg_metrics['precision'][i],
                avg_metrics['recall'][i],
                avg_metrics['f1_score'][i],
            ]
            values += values[:1]  # Close the loop
            
            # Plot the model's metrics with more transparent fill
            self.ax3.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=model_colors[i])
            self.ax3.fill(angles, values, alpha=0.3, color=model_colors[i])
            
            # Add value labels with smaller font and rotation
            for angle, value in zip(angles[:-1], values[:-1]):
                self.ax3.text(angle, value + 0.02, f'{value:.2f}', 
                            ha='center', va='center', fontsize=fontsize-2, rotation=23, color="#222222")
        
        self.ax3.set_title('Model Performance Comparison (Radar Chart)', fontsize=fontsize*1.5)
        # Move legend outside the plot to avoid overlap
        self.ax3.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=fontsize*1.3, facecolor=facecolors[1])

        leg = self.ax3.legend()

        # change the line width for the legend
        for line in leg.get_lines():
            line.set_linewidth(fontsize)
        
        # Update training loss plot
        if self.training_losses:  # Use the stored training losses
            # Filter out models with no losses
            valid_losses = {}
            for model_name, losses in self.training_losses.items():
                if model_name == "d_lstm_mlp" and isinstance(losses, dict):
                    # Handle D_LSTM_MLP special case
                    if losses.get('lstm') and isinstance(losses['lstm'], list) and len(losses['lstm']) > 0:
                        valid_losses[f"{model_name}-lstm"] = losses['lstm']
                    if losses.get('mlp') and isinstance(losses['mlp'], list) and len(losses['mlp']) > 0:
                        valid_losses[f"{model_name}-mlp"] = losses['mlp']
                elif isinstance(losses, list) and len(losses) > 0:
                    valid_losses[model_name] = losses
            
            if valid_losses:  # Only plot if we have valid losses
                # Use a colormap for different models
                colors = plt.cm.Set3(np.linspace(0, 1, len(valid_losses)))
                
                # Plot each model's loss curve
                for (model_name, losses), color in zip(valid_losses.items(), colors):
                    try:
                        self.ax4.plot(losses, label=model_name, color=color, linewidth=2)
                        
                        # Add final loss value annotation
                        final_loss = losses[-1]
                        self.ax4.annotate(f'{final_loss:.4f}',
                                        xy=(len(losses)-1, final_loss),
                                        xytext=(len(losses)-1, final_loss*1.1),
                                        arrowprops=dict(facecolor='#333333', shrink=0.05, width=1, headwidth=fontsize, headlength=fontsize),
                                        fontsize=8, rotation=30)
                    except Exception as e:
                        print(f"Error plotting losses for {model_name}: {str(e)}")
                        continue

                self.ax4.set_title('Training Loss Curves for All Models', fontsize=fontsize*1.5)
                self.ax4.set_facecolor(facecolors[0])
                self.ax4.set_xlabel('Epoch', fontsize=fontsize*1.3)
                self.ax4.set_ylabel('Loss', fontsize=fontsize*1.3)
                self.ax4.grid(True, linestyle='--', alpha=0.7)

                # Only add legend if we have valid plots
                if self.ax4.get_lines():
                    self.ax4.legend(bbox_to_anchor=(0.7, 1), loc='upper left', fontsize=fontsize)
                    leg = self.ax4.legend()
                    # change the line width for the legend
                    for line in leg.get_lines():
                        line.set_linewidth(fontsize)
                # Adjust layout to prevent label cutoff
                self.fig3.tight_layout()
            else:
                # Display message if no valid losses
                self.ax4.text(0.5, 0.5, 'No training losses recorded for any model',
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=self.ax4.transAxes,
                            fontsize=12)
                self.ax4.set_title('Training Loss Curves', fontsize=14)
                self.ax4.set_xticks([])
                self.ax4.set_yticks([])

        # Update training time plot
        if any(training_times):  # Only plot if we have timing data
            # Sort models by training time
            sorted_indices = np.argsort(training_times)
            sorted_models = [models[i] for i in sorted_indices]
            sorted_times = [training_times[i] for i in sorted_indices]

            # Create horizontal bar chart for training times
            bars = self.ax5.barh(sorted_models, sorted_times, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
            
            # Add value labels on the bars
            for bar in bars:
                width = bar.get_width()
                self.ax5.text(width, bar.get_y() + bar.get_height()/2.,
                            f'{width:.1f}s',
                            ha='left', va='center', fontsize=10, color='#222222',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

            self.ax5.set_title('Model Training Time Comparison', fontsize=12)
            self.ax5.set_xlabel('Training Time (seconds)', fontsize=10)
            self.ax5.grid(True, linestyle='--', alpha=0.7)
            self.ax5.set_facecolor('#f2f2f2')
            
            # Adjust layout
            self.fig4.tight_layout()
        else:
            # Display message if no timing data
            self.ax5.text(0.5, 0.5, 'No training time data available',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=self.ax5.transAxes,
                        fontsize=12)
            self.ax5.set_title('Training Time Comparison', fontsize=12)
            self.ax5.set_xticks([])
            self.ax5.set_yticks([])

        # Update the results text display
        def update_results_text():
            self.results_text.delete("1.0", ctk.END)
            for model_name, metrics in self.results.items():
                self.results_text.insert(ctk.END, f"\nResults for {model_name}:\n")
                avg_metrics = {
                    'accuracy': np.mean(metrics['accuracy']),
                    'recall': np.mean(metrics['recall']),
                    'precision': np.mean(metrics['precision']),
                    'f1_score': np.mean(metrics['f1_score'])
                }
                self.results_text.insert(ctk.END, f"Average Accuracy: {avg_metrics['accuracy']:.4f}\n")
                self.results_text.insert(ctk.END, f"Average Recall: {avg_metrics['recall']:.4f}\n")
                self.results_text.insert(ctk.END, f"Average Precision: {avg_metrics['precision']:.4f}\n")
                self.results_text.insert(ctk.END, f"Average F1-Score: {avg_metrics['f1_score']:.4f}\n")
                self.results_text.insert(ctk.END, f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")
                self.results_text.insert(ctk.END, "-" * 50 + "\n")

        update_results_text()

        # Draw all canvases
        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()
        self.canvas4.draw()

        # Save charts
        graph_dir = self.config.get("data.graph_dir")
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        self.fig1.savefig(os.path.join(graph_dir, "performance_metrics.png"))
        self.fig2.savefig(os.path.join(graph_dir, "model_comparison.png"))
        if valid_losses:  # Only save if we have valid losses
            self.fig3.savefig(os.path.join(graph_dir, "all_models_training_loss.png"))
        if any(training_times):  # Only save if we have timing data
            self.fig4.savefig(os.path.join(graph_dir, "training_time_comparison.png"))
            
            # Remove custom plotting code for D_LSTM_MLP components since we're using base_model's plot_train_loss
            # The individual component plots are now saved in app.py using the base_model's plot_train_loss method

        cm_dir = self.config.get("data.cm_dir")
        if not os.path.exists(cm_dir):
            os.makedirs(cm_dir)
        # Save individual confusion matrix heatmaps
        for (model_name, cm) in self.confusion_matrices.items():
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Calculate percentages
            cm_percentages = cm / np.sum(cm) * 100
            
            # Create custom annotations combining count and percentage
            annot = np.empty_like(cm, dtype=object)
            for i in range(len(cm)):
                for j in range(len(cm)):
                    annot[i, j] = f"{cm[i, j]}\n({cm_percentages[i, j]:.1f}%)"
            
            # Create heatmap without default annotations
            sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax, cbar=False)
            
            # Add custom annotations with dynamic text color
            for i in range(len(cm)):
                for j in range(len(cm)):
                    # Calculate background color intensity
                    color_intensity = cm[i, j] / cm.max()
                    # Use white text for dark backgrounds, black for light backgrounds
                    text_color = '#fafafa' if color_intensity > 0.5 else '#222222'
                    
                    ax.text(j + 0.5, i + 0.5, annot[i, j],
                           ha='center', va='center',
                           color=text_color,
                           fontsize=fontsize)
            
            ax.set_title(f"Confusion Matrix ({model_name})")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            fig.tight_layout()
            fig.savefig(os.path.join(cm_dir, f"confusion_matrix_{model_name}.png"))
            plt.close(fig)

    def toggle_kfold_controls(self):
        """Toggle visibility of K-fold controls based on checkbox state."""
        if self.use_kfold_var.get():
            self.kfold_frame.pack(pady=5, fill=ctk.X)
            self.kfold_entry_frame.pack(pady=5, fill=ctk.X)
        else:
            self.kfold_frame.pack_forget()
            self.kfold_entry_frame.pack_forget()

    def run_models(self):
        """Run the selected models and update the display."""
        # Update configuration
        self.config.set('models.train_percentage', int(self.train_split.get()))
        n_folds = int(self.k_fold.get()) if self.use_kfold_var.get() else 1
        subsample_rate = round(self.subsample_rate.get() / 100.0, 3)

        # Clear previous results
        self.results_text.delete("1.0", ctk.END)
        self.results.clear()
        self.confusion_matrices.clear()
        self.training_losses.clear()  # Clear previous training losses

        # Run selected models
        for model_name, var in self.model_vars.items():
            if var.get():
                try:
                    self.results_text.insert(ctk.END, f"\nRunning {model_name}...\n")
                    if not self.use_kfold_var.get():
                        self.results_text.insert(ctk.END, "Using single train/test split...\n")
                    self.root.update()
                    
                    metrics = self.predictor.run(
                        model_name,
                        subsample_rate,
                        n_folds,
                        debug_mode=self.debug_mode,
                        use_feature_engineering=self.feature_eng_var.get(),
                    )

                    self.results[model_name] = metrics

                    # Store confusion matrix if available
                    if 'confusion_matrix' in metrics:
                        self.confusion_matrices[model_name] = metrics['confusion_matrix']

                    # Store training losses if available
                    if 'training_losses' in metrics and metrics['training_losses']:
                        self.training_losses[model_name] = metrics['training_losses']

                    # Display average results
                    avg_metrics = {
                        'accuracy': np.mean(metrics['accuracy']),
                        'recall': np.mean(metrics['recall']),
                        'precision': np.mean(metrics['precision']),
                        'f1_score': np.mean(metrics['f1_score'])
                    }

                    self.results_text.insert(ctk.END, f"Average Accuracy: {avg_metrics['accuracy']:.4f}\n")
                    self.results_text.insert(ctk.END, f"Average Recall: {avg_metrics['recall']:.4f}\n")
                    self.results_text.insert(ctk.END, f"Average Precision: {avg_metrics['precision']:.4f}\n")
                    self.results_text.insert(ctk.END, f"Average F1-Score: {avg_metrics['f1_score']:.4f}\n")
                    self.results_text.insert(ctk.END, f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")
                except Exception as e:
                    self.results_text.insert(ctk.END, f"Error: {str(e)}\n")

        # Update charts including training losses
        self.update_charts()
    
    def run(self):
        """Start the GUI application."""
        print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        self.root.mainloop() 

    # def evaluate_test_data(self):
    #     """Evaluate models on test data and save metrics and charts."""
    #     try:
    #         # Clear previous results
    #         self.results_text.delete("1.0", ctk.END)
    #         self.results.clear()
    #         self.confusion_matrices.clear()
            
    #         # Load test data
    #         test_path = self.config.get("data.test_path")
    #         self.results_text.insert(ctk.END, f"\nLoading test data from {test_path}...\n")
    #         self.root.update()
            
    #         # Get test data
    #         test_data, test_labels = self.predictor.data_repo.load_test_data()
            
    #         # Run evaluation for each selected model
    #         for model_name, var in self.model_vars.items():
    #             if var.get():
    #                 try:
    #                     self.results_text.insert(ctk.END, f"\nEvaluating {model_name} on test data...\n")
    #                     self.root.update()
                        
    #                     # Create and train model
    #                     model = self.predictor.create_model(model_name)
    #                     model.train(self.predictor.data_repo.X_train, self.predictor.data_repo.y_train)
                        
    #                     # Evaluate on test data
    #                     accuracy, recall, precision = model.evaluate(test_data, test_labels)
                        
    #                     # Get predictions for confusion matrix
    #                     predictions = model.predict(test_data)
    #                     predictions = (predictions > model.find_best_threshold(test_labels, predictions)).astype(int)
    #                     cm = confusion_matrix(test_labels, predictions)
                        
    #                     # Store results
    #                     self.results[model_name] = {
    #                         'accuracy': [accuracy],
    #                         'recall': [recall],
    #                         'precision': [precision],
    #                         'f1_score': [2 * (recall * precision) / (recall + precision)],
    #                         'confusion_matrix': cm
    #                     }
    #                     self.confusion_matrices[model_name] = cm
                        
    #                     # Display results
    #                     self.results_text.insert(ctk.END, f"Test Accuracy: {accuracy:.4f}\n")
    #                     self.results_text.insert(ctk.END, f"Test Recall: {recall:.4f}\n")
    #                     self.results_text.insert(ctk.END, f"Test Precision: {precision:.4f}\n")
    #                     self.results_text.insert(ctk.END, f"Test F1-Score: {self.results[model_name]['f1_score'][0]:.4f}\n")
    #                     self.results_text.insert(ctk.END, f"Confusion Matrix:\n{cm}\n")
    #                     self.results_text.insert(ctk.END, "-" * 50 + "\n")
                        
    #                     # Save model's training loss curve
    #                     model.plot_train_loss()
                        
    #                 except Exception as e:
    #                     self.results_text.insert(ctk.END, f"Error evaluating {model_name}: {str(e)}\n")
            
    #         # Update and save charts
    #         self.update_charts()
            
    #         # Save test results to file
    #         results_dir = os.path.join(self.config.get("data.graph_dir"), "test_results")
    #         if not os.path.exists(results_dir):
    #             os.makedirs(results_dir)
            
    #         # Save metrics to CSV
    #         metrics_df = pd.DataFrame({
    #             'Model': [],
    #             'Accuracy': [],
    #             'Recall': [],
    #             'Precision': [],
    #             'F1-Score': []
    #         })
            
    #         for model_name, metrics in self.results.items():
    #             metrics_df = pd.concat([metrics_df, pd.DataFrame({
    #                 'Model': [model_name],
    #                 'Accuracy': [metrics['accuracy'][0]],
    #                 'Recall': [metrics['recall'][0]],
    #                 'Precision': [metrics['precision'][0]],
    #                 'F1-Score': [metrics['f1_score'][0]]
    #             })], ignore_index=True)
            
    #         metrics_df.to_csv(os.path.join(results_dir, "test_metrics.csv"), index=False)
            
    #         # Save charts with test-specific names
    #         self.fig1.savefig(os.path.join(results_dir, "test_performance_metrics.png"))
    #         self.fig2.savefig(os.path.join(results_dir, "test_model_comparison.png"))
            
    #         # Save individual confusion matrices
    #         cm_dir = os.path.join(results_dir, "confusion_matrices")
    #         if not os.path.exists(cm_dir):
    #             os.makedirs(cm_dir)
            
    #         for model_name, cm in self.confusion_matrices.items():
    #             fig, ax = plt.subplots(figsize=(6, 6))
                
    #             # Calculate percentages
    #             cm_percentages = cm / np.sum(cm) * 100
                
    #             # Create custom annotations combining count and percentage
    #             annot = np.empty_like(cm, dtype=object)
    #             for i in range(len(cm)):
    #                 for j in range(len(cm)):
    #                     annot[i, j] = f"{cm[i, j]}\n({cm_percentages[i, j]:.1f}%)"
                
    #             # Create heatmap without default annotations
    #             sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax, cbar=False)
                
    #             # Add custom annotations with dynamic text color
    #             for i in range(len(cm)):
    #                 for j in range(len(cm)):
    #                     # Calculate background color intensity
    #                     color_intensity = cm[i, j] / cm.max()
    #                     # Use white text for dark backgrounds, black for light backgrounds
    #                     text_color = 'white' if color_intensity > 0.5 else 'black'
                        
    #                     ax.text(j + 0.5, i + 0.5, annot[i, j],
    #                            ha='center', va='center',
    #                            color=text_color,
    #                            fontsize=9,
    #                            fontweight='bold')
                
    #             ax.set_title(f"Test Confusion Matrix ({model_name})")
    #             ax.set_xlabel("Predicted")
    #             ax.set_ylabel("Actual")
    #             fig.tight_layout()
    #             fig.savefig(os.path.join(cm_dir, f"test_confusion_matrix_{model_name}.png"))
    #             plt.close(fig)
            
    #         self.results_text.insert(ctk.END, f"\nTest evaluation results saved to {results_dir}\n")
            
    #     except Exception as e:
    #         self.results_text.insert(ctk.END, f"Error in test evaluation: {str(e)}\n") 
