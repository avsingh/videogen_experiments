cat > finetune_wan80gb.py << 'EOF'
#!/usr/bin/env python3
"""
WAN 2.2 Fine-tuning on Lambda Labs A100 80GB
Optimized for Google Photos video dataset
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, TaskType
import wandb
import os
from pathlib import Path
from tqdm import tqdm
import json

from wan_dataset import GooglePhotosVideoDataset

# Import WAN 2.2 model components
# TODO: Update these imports based on actual WAN2.2 repo structure
# from wan.models import WANVideoModel
# from wan.config import WANConfig

class WANFineTuner:
    """Fine-tuner for WAN 2.2 on A100 80GB"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize Accelerator for efficient training
        self.accelerator = Accelerator(
            mixed_precision=config.get('mixed_precision', 'fp16'),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
            log_with='wandb' if config.get('use_wandb', True) else None,
        )
        
        if self.accelerator.is_main_process:
            print("="*60)
            print("WAN 2.2 Fine-Tuning on A100 80GB")
            print("="*60)
            print(f"Mixed precision: {config.get('mixed_precision', 'fp16')}")
            print(f"Gradient accumulation: {config.get('gradient_accumulation_steps', 4)}")
            print(f"Effective batch size: {config['batch_size'] * config.get('gradient_accumulation_steps', 4)}")
        
        # Load WAN 2.2 model
        print("\nLoading WAN 2.2 model...")
        # TODO: Load actual WAN model
        # self.model = WANVideoModel.from_pretrained(config['model_path'])
        
        # Enable memory optimizations
        if hasattr(self.model, 'enable_gradient_checkpointing'):
            print("Enabling gradient checkpointing...")
            self.model.enable_gradient_checkpointing()
        
        # Apply LoRA for parameter-efficient fine-tuning
        if config.get('use_lora', True):
            print(f"\nApplying LoRA:")
            print(f"  Rank: {config['lora_rank']}")
            print(f"  Alpha: {config['lora_alpha']}")
            
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config['lora_rank'],
                lora_alpha=config.get('lora_alpha', 64),
                lora_dropout=config.get('lora_dropout', 0.1),
                target_modules=config.get('lora_target_modules', 
                    ["q_proj", "k_proj", "v_proj", "out_proj"]),
                bias="none",
            )
            
            # self.model = get_peft_model(self.model, lora_config)
            
            # if self.accelerator.is_main_process:
            #     self.model.print_trainable_parameters()
        
        # Create dataset
        print(f"\nLoading dataset from {config['data_dir']}...")
        self.train_dataset = GooglePhotosVideoDataset(
            config['data_dir'],
            num_frames=config.get('num_frames', 240),
            size=config.get('video_size', (512, 512))
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            prefetch_factor=2,
        )
        
        # Optimizer
        print("\nSetting up optimizer...")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.999),
            weight_decay=config.get('weight_decay', 0.01),
        )
        
        # Learning rate scheduler
        num_training_steps = len(self.train_loader) * config['num_epochs']
        num_warmup_steps = int(0.1 * num_training_steps)
        
        from transformers import get_cosine_schedule_with_warmup
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        print(f"  Total training steps: {num_training_steps}")
        print(f"  Warmup steps: {num_warmup_steps}")
        
        # Prepare with Accelerator
        (self.model, self.optimizer, self.train_loader, 
         self.lr_scheduler) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.lr_scheduler
        )
        
        # Initialize WandB
        if self.accelerator.is_main_process and config.get('use_wandb', True):
            wandb.init(
                project=config.get('wandb_project', 'wan-finetune'),
                name=config.get('run_name', 'google-photos'),
                config=config
            )
            print(f"\nWandB initialized: {wandb.run.get_url()}")
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Save config
        if self.accelerator.is_main_process:
            config_path = Path(config['output_dir']) / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"\nConfig saved to: {config_path}")
        
        print("\n" + "="*60)
        print("Setup complete! Starting training...")
        print("="*60 + "\n")
    
    def train(self):
        """Main training loop"""
        self.model.train()
        global_step = 0
        best_loss = float('
