import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import PolygonColorUNet, count_parameters
from dataset import create_dataloaders, COLOR_NAMES
from utils import calculate_iou, save_checkpoint, load_checkpoint

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                config=vars(config),
                name=config.run_name
            )

        # Create model
        self.model = PolygonColorUNet(n_colors=len(COLOR_NAMES)).to(self.device)
        print(f"Model has {count_parameters(self.model):,} trainable parameters")

        # Create data loaders
        self.train_loader, self.val_loader = create_dataloaders(
            data_root=config.data_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )

        # Loss function and optimizer
        self.criterion = nn.MSELoss()  # Can also try L1Loss or combination
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01
        )

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.best_iou = 0.0
        self.start_epoch = 0

        # Load checkpoint if exists
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            inputs = batch["input_polygon"].to(self.device)
            colors = batch["colour_onehot"].to(self.device)
            targets = batch["output_image"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs, colors)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update running loss
            running_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log to wandb
            if self.config.use_wandb and batch_idx % self.config.log_interval == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": self.optimizer.param_groups[0]['lr']
                })

        avg_loss = running_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        running_iou = 0.0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                # Move data to device
                inputs = batch["input_polygon"].to(self.device)
                colors = batch["colour_onehot"].to(self.device)
                targets = batch["output_image"].to(self.device)

                # Forward pass
                outputs = self.model(inputs, colors)
                loss = self.criterion(outputs, targets)

                # Calculate IoU
                iou = calculate_iou(outputs, targets)

                running_loss += loss.item()
                running_iou += iou.item()

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "iou": f"{iou.item():.4f}"
                })

        avg_loss = running_loss / len(self.val_loader)
        avg_iou = running_iou / len(self.val_loader)

        self.val_losses.append(avg_loss)
        self.val_ious.append(avg_iou)

        return avg_loss, avg_iou

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ious': self.val_ious,
            'best_iou': self.best_iou,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'last_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

            if self.config.use_wandb:
                wandb.save(best_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_ious = checkpoint.get('val_ious', [])
        self.best_iou = checkpoint.get('best_iou', 0.0)

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config.epochs} epochs...")

        for epoch in range(self.start_epoch, self.config.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_iou = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Check if best model
            is_best = val_iou > self.best_iou
            if is_best:
                self.best_iou = val_iou
                print(f"New best model! IoU: {val_iou:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch, is_best)

            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss,
                    "val/iou": val_iou,
                    "val/best_iou": self.best_iou
                })

            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        print(f"\nTraining completed! Best IoU: {self.best_iou:.4f}")

        # Save final checkpoint
        self.save_checkpoint(self.config.epochs - 1, False)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Polygon Color UNet')

    # Data arguments
    parser.add_argument('--data_root', type=str, default='data/dataset',
                        help='Root directory of the dataset')

    # Model arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')

    # Training arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log to wandb every N batches')

    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--project_name', type=str, default='polygon-color-unet',
                        help='Wandb project name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Wandb run name')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()
