import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import time
import logging
from pathlib import Path
import warnings

class TrainingDifficultyTracker:
    """
    Comprehensive tracker for analyzing DeepAR training difficulties
    """
    
    def __init__(self, model, save_dir="training_analysis", experiment_name="deepar_exp"):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.experiment_name = experiment_name
        
        # Initialize tracking dictionaries
        self.metrics = defaultdict(list)
        self.gradient_stats = defaultdict(list)
        self.layer_stats = defaultdict(lambda: defaultdict(list))
        self.training_events = []
        self.epoch_times = []
        
        # Training state
        self.current_epoch = 0
        self.start_time = None
        self.last_stable_loss = float('inf')
        self.loss_spikes = 0
        self.nan_incidents = 0
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup detailed logging for training events"""
        log_file = self.save_dir / f"{self.experiment_name}_training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def start_epoch(self):
        """Call at the beginning of each epoch"""
        self.current_epoch += 1
        self.start_time = time.time()
        self.logger.info(f"Starting epoch {self.current_epoch}")
    
    def end_epoch(self):
        """Call at the end of each epoch"""
        if self.start_time:
            epoch_time = time.time() - self.start_time
            self.epoch_times.append(epoch_time)
            self.metrics['epoch_time'].append(epoch_time)
    
    def track_gradients(self, step=None):
        """
        Track comprehensive gradient statistics
        Call this after loss.backward() but before optimizer.step()
        """
        total_norm = 0.0
        param_count = 0
        zero_grad_count = 0
        layer_norms = {}
        grad_to_param_ratios = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Calculate parameter-wise statistics
                param_norm = param.grad.data.norm(2)
                param_size = param.numel()
                
                # Track layer-wise statistics
                layer_name = name.split('.')[0]  # Get base layer name
                if layer_name not in layer_norms:
                    layer_norms[layer_name] = []
                layer_norms[layer_name].append(param_norm.item())
                
                # Gradient-to-parameter ratio
                if param.data.norm(2) > 0:
                    ratio = param_norm / param.data.norm(2)
                    grad_to_param_ratios[name] = ratio.item()
                
                # Count zero gradients
                if param_norm < 1e-10:
                    zero_grad_count += 1
                
                total_norm += param_norm ** 2
                param_count += param_size
                
                # Store per-layer statistics
                self.layer_stats[name]['grad_norm'].append(param_norm.item())
                self.layer_stats[name]['param_norm'].append(param.data.norm(2).item())
                self.layer_stats[name]['grad_to_param_ratio'].append(
                    grad_to_param_ratios.get(name, 0.0)
                )
        
        # Calculate aggregate statistics
        total_norm = total_norm ** 0.5
        
        # Convert to float if it's a tensor
        if hasattr(total_norm, 'item'):
            total_norm_value = total_norm.item()
        else:
            total_norm_value = float(total_norm)
        
        # Store gradient statistics
        self.gradient_stats['total_norm'].append(total_norm_value)
        self.gradient_stats['zero_grad_percentage'].append(
            (zero_grad_count / len(list(self.model.parameters()))) * 100
        )
        
        # Layer-wise gradient norm statistics
        for layer_name, norms in layer_norms.items():
            self.gradient_stats[f'{layer_name}_max_norm'].append(max(norms))
            self.gradient_stats[f'{layer_name}_min_norm'].append(min(norms))
            self.gradient_stats[f'{layer_name}_mean_norm'].append(np.mean(norms))
            self.gradient_stats[f'{layer_name}_std_norm'].append(np.std(norms))
        
        # Check for gradient explosion/vanishing
        if total_norm_value > 10.0:  # Threshold for explosion
            self.training_events.append({
                'epoch': self.current_epoch,
                'step': step,
                'event': 'gradient_explosion',
                'value': total_norm_value
            })
            self.logger.warning(f"Gradient explosion detected: {total_norm_value:.4f}")
        
        if total_norm_value < 1e-6:  # Threshold for vanishing
            self.training_events.append({
                'epoch': self.current_epoch,
                'step': step,
                'event': 'gradient_vanishing',
                'value': total_norm_value
            })
            self.logger.warning(f"Gradient vanishing detected: {total_norm_value:.4f}")
        
        return total_norm_value
    
    def track_loss(self, train_loss, val_loss=None, step=None):
        """Track loss statistics and detect anomalies"""
        self.metrics['train_loss'].append(train_loss)
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        
        # Check for NaN/Inf
        if np.isnan(train_loss) or np.isinf(train_loss):
            self.nan_incidents += 1
            self.training_events.append({
                'epoch': self.current_epoch,
                'step': step,
                'event': 'nan_loss',
                'value': train_loss
            })
            self.logger.error(f"NaN/Inf loss detected: {train_loss}")
        
        # Check for loss spikes
        if len(self.metrics['train_loss']) > 10:
            recent_losses = self.metrics['train_loss'][-10:]
            current_loss = train_loss
            median_loss = np.median(recent_losses[:-1])
            
            if current_loss > median_loss * 2:  # 2x spike threshold
                self.loss_spikes += 1
                self.training_events.append({
                    'epoch': self.current_epoch,
                    'step': step,
                    'event': 'loss_spike',
                    'value': current_loss / median_loss
                })
                self.logger.warning(f"Loss spike detected: {current_loss:.4f} vs median {median_loss:.4f}")
    
    def track_learning_rate(self, optimizer):
        """Track learning rate changes"""
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.metrics[f'learning_rate_group_{i}'].append(lr)
    
    def track_model_weights(self):
        """Track weight statistics"""
        weight_stats = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weight_norm = param.data.norm(2).item()
                weight_stats[f'{name}_norm'] = weight_norm
                weight_stats[f'{name}_mean'] = param.data.mean().item()
                weight_stats[f'{name}_std'] = param.data.std().item()
        
        self.metrics['weight_stats'].append(weight_stats)
    
    def calculate_gradient_flow_metrics(self):
        """Calculate advanced gradient flow metrics"""
        if len(self.gradient_stats['total_norm']) < 2:
            return
        
        # Gradient norm variance
        recent_norms = self.gradient_stats['total_norm'][-50:]  # Last 50 steps
        grad_variance = np.var(recent_norms)
        self.metrics['gradient_variance'].append(grad_variance)
        
        # Effective gradient magnitude
        for name in self.layer_stats:
            if len(self.layer_stats[name]['grad_norm']) > 0:
                effective_grad = np.mean(self.layer_stats[name]['grad_to_param_ratio'][-10:])
                self.metrics[f'{name}_effective_gradient'].append(effective_grad)
    
    def assess_training_stability(self):
        """Assess overall training stability"""
        stability_score = 0
        
        # Loss stability (lower variance = more stable)
        if len(self.metrics['train_loss']) > 50:
            recent_losses = self.metrics['train_loss'][-50:]
            loss_cv = np.std(recent_losses) / np.mean(recent_losses)
            stability_score += min(loss_cv, 1.0)  # Cap at 1.0
        
        # Gradient stability
        if len(self.gradient_stats['total_norm']) > 50:
            recent_grads = self.gradient_stats['total_norm'][-50:]
            grad_cv = np.std(recent_grads) / (np.mean(recent_grads) + 1e-8)
            stability_score += min(grad_cv, 1.0)
        
        # Event-based penalties
        stability_score += self.loss_spikes * 0.1
        stability_score += self.nan_incidents * 0.5
        
        self.metrics['stability_score'].append(stability_score)
        return stability_score
    
    def generate_training_report(self):
        """Generate comprehensive training difficulty report"""
        report = {
            'experiment_name': self.experiment_name,
            'total_epochs': self.current_epoch,
            'total_training_time': sum(self.epoch_times),
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'training_events': {
                'loss_spikes': self.loss_spikes,
                'nan_incidents': self.nan_incidents,
                'gradient_explosions': len([e for e in self.training_events if e['event'] == 'gradient_explosion']),
                'gradient_vanishing': len([e for e in self.training_events if e['event'] == 'gradient_vanishing'])
            },
            'gradient_statistics': {
                'max_gradient_norm': max(self.gradient_stats['total_norm']) if self.gradient_stats['total_norm'] else 0,
                'min_gradient_norm': min(self.gradient_stats['total_norm']) if self.gradient_stats['total_norm'] else 0,
                'mean_gradient_norm': np.mean(self.gradient_stats['total_norm']) if self.gradient_stats['total_norm'] else 0,
                'gradient_norm_std': np.std(self.gradient_stats['total_norm']) if self.gradient_stats['total_norm'] else 0
            },
            'stability_assessment': {
                'final_stability_score': self.assess_training_stability(),
                'training_completed': self.nan_incidents == 0
            }
        }
        
        return report
    
    def save_data(self):
        """Save all tracking data"""
        # Save metrics
        metrics_df = pd.DataFrame({
            k: v for k, v in self.metrics.items() 
            if isinstance(v, list) and len(v) > 0 and not isinstance(v[0], dict)
        })
        metrics_df.to_csv(self.save_dir / f"{self.experiment_name}_metrics.csv", index=False)
        
        # Save gradient statistics
        grad_df = pd.DataFrame(self.gradient_stats)
        grad_df.to_csv(self.save_dir / f"{self.experiment_name}_gradients.csv", index=False)
        
        # Save layer statistics
        for layer_name, stats in self.layer_stats.items():
            layer_df = pd.DataFrame(stats)
            layer_df.to_csv(self.save_dir / f"{self.experiment_name}_{layer_name}_stats.csv", index=False)
        
        # Save training events
        events_df = pd.DataFrame(self.training_events)
        events_df.to_csv(self.save_dir / f"{self.experiment_name}_events.csv", index=False)
        
        # Save comprehensive report
        report = self.generate_training_report()
        with open(self.save_dir / f"{self.experiment_name}_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Training data saved to {self.save_dir}")
    
    def plot_training_analysis(self):
        """Generate comprehensive training analysis plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss curves
        if 'train_loss' in self.metrics:
            axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss', alpha=0.7)
            if 'val_loss' in self.metrics:
                axes[0, 0].plot(self.metrics['val_loss'], label='Val Loss', alpha=0.7)
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].set_yscale('log')
        
        # Gradient norms
        if 'total_norm' in self.gradient_stats:
            axes[0, 1].plot(self.gradient_stats['total_norm'], alpha=0.7)
            axes[0, 1].set_title('Gradient Norms')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Gradient Norm')
            axes[0, 1].set_yscale('log')
        
        # Gradient norm distribution
        if 'total_norm' in self.gradient_stats:
            axes[0, 2].hist(self.gradient_stats['total_norm'], bins=50, alpha=0.7)
            axes[0, 2].set_title('Gradient Norm Distribution')
            axes[0, 2].set_xlabel('Gradient Norm')
            axes[0, 2].set_ylabel('Frequency')
        
        # Layer-wise gradient norms (heatmap)
        layer_data = []
        layer_names = []
        for layer_name in self.layer_stats:
            if 'grad_norm' in self.layer_stats[layer_name]:
                layer_data.append(self.layer_stats[layer_name]['grad_norm'][-100:])  # Last 100 steps
                layer_names.append(layer_name)
        
        if layer_data:
            # Pad sequences to same length
            max_len = max(len(seq) for seq in layer_data)
            layer_data_padded = [seq + [0] * (max_len - len(seq)) for seq in layer_data]
            
            im = axes[1, 0].imshow(layer_data_padded, aspect='auto', cmap='viridis')
            axes[1, 0].set_title('Layer-wise Gradient Norms')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Layer')
            axes[1, 0].set_yticks(range(len(layer_names)))
            axes[1, 0].set_yticklabels(layer_names)
            plt.colorbar(im, ax=axes[1, 0])
        
        # Training events timeline
        if self.training_events:
            events_df = pd.DataFrame(self.training_events)
            event_counts = events_df['event'].value_counts()
            axes[1, 1].bar(event_counts.index, event_counts.values)
            axes[1, 1].set_title('Training Events')
            axes[1, 1].set_xlabel('Event Type')
            axes[1, 1].set_ylabel('Count')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # Stability score
        if 'stability_score' in self.metrics:
            axes[1, 2].plot(self.metrics['stability_score'], alpha=0.7)
            axes[1, 2].set_title('Training Stability Score')
            axes[1, 2].set_xlabel('Step')
            axes[1, 2].set_ylabel('Stability Score (lower = better)')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{self.experiment_name}_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()