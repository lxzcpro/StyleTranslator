"""Training script for style detection models with WandB logging"""
import os
import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from dataset import StyleDataset, create_data_splits
from model import StyleDetector
import wandb


def load_config(config_path: str) -> dict:
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_model(language: str, config_path: str = 'config.yaml', **overrides):
    """Train style detection model"""
    # Load and merge configs
    cfg = load_config(config_path)
    model_cfg = {**cfg[language], **cfg['training']}
    model_cfg.update({k: v for k, v in overrides.items() if v is not None})
    
    print(f"\nTraining {language.upper()} Style Detector")
    print(f"Config: {model_cfg}\n")
    
    # Create datasets
    train_ds, val_ds, test_ds = create_data_splits(
        csv_path=model_cfg['dataset_path'],
        tokenizer_name=model_cfg['model_name'],
        language_filter=language,
        max_length=model_cfg['max_length'],
        train_ratio=cfg['data']['train_ratio'],
        val_ratio=cfg['data']['val_ratio']
    )
    
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Classes ({train_ds.num_classes}): {train_ds.get_style_labels()}\n")
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=model_cfg['batch_size'], 
                             shuffle=True, num_workers=model_cfg['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=model_cfg['batch_size'], 
                           shuffle=False, num_workers=model_cfg['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=model_cfg['batch_size'], 
                            shuffle=False, num_workers=model_cfg['num_workers'], pin_memory=True)
    
    # Initialize model
    model = StyleDetector(
        model_name=model_cfg['model_name'],
        num_classes=train_ds.num_classes,
        learning_rate=model_cfg['learning_rate'],
        warmup_steps=model_cfg['warmup_steps'],
        dropout_rate=model_cfg['dropout_rate'],
        freeze_bert_layers=model_cfg['freeze_bert_layers']
    )
    
    # Setup callbacks
    checkpoint_dir = f"checkpoints/{language}_style_detector"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}',
            monitor=cfg['callbacks']['checkpoint']['monitor'],
            mode=cfg['callbacks']['checkpoint']['mode'],
            save_top_k=cfg['callbacks']['checkpoint']['save_top_k'],
            save_last=cfg['callbacks']['checkpoint']['save_last']
        ),
        EarlyStopping(
            monitor=cfg['callbacks']['early_stopping']['monitor'],
            patience=cfg['callbacks']['early_stopping']['patience'],
            mode=cfg['callbacks']['early_stopping']['mode']
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Setup WandB logger
    wandb_cfg = cfg['wandb']
    logger = WandbLogger(
        project=overrides.get('wandb_project') or wandb_cfg['project'],
        entity=overrides.get('wandb_entity') or wandb_cfg.get('entity'),
        name=overrides.get('wandb_name') or f"{language}_style_detector",
        tags=(overrides.get('wandb_tags') or wandb_cfg.get('tags', [])) + [language],
        log_model=True
    )
    
    # Log config
    logger.experiment.config.update({
        **model_cfg,
        'language': language,
        'train_size': len(train_ds),
        'val_size': len(val_ds),
        'test_size': len(test_ds),
        'num_classes': train_ds.num_classes,
    })
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=model_cfg['epochs'],
        accelerator=model_cfg['accelerator'],
        devices=model_cfg['devices'],
        precision=model_cfg['precision'],
        callbacks=callbacks,
        logger=logger,
        val_check_interval=model_cfg['val_check_interval'],
        log_every_n_steps=model_cfg['log_every_n_steps'],
        gradient_clip_val=model_cfg['gradient_clip_val'],
        accumulate_grad_batches=model_cfg['accumulate_grad_batches']
    )
    
    # Train and test
    trainer.fit(model, train_loader, val_loader)
    test_results = trainer.test(model, test_loader)
    
    # Save final model
    final_path = f"models/{language}_style_detector_final.ckpt"
    os.makedirs("models", exist_ok=True)
    trainer.save_checkpoint(final_path)
    
    # Save model artifact to WandB
    artifact = wandb.Artifact(f"{language}_style_detector", type='model')
    artifact.add_file(final_path)
    logger.experiment.log_artifact(artifact)
    
    wandb.finish()
    print(f"\nModel saved: {final_path}")
    
    return model, trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', required=True, choices=['chinese', 'english'])
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--accelerator', type=str)
    parser.add_argument('--devices', type=int)
    parser.add_argument('--precision', type=str)
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--wandb_entity', type=str)
    parser.add_argument('--wandb_name', type=str)
    parser.add_argument('--wandb_tags', nargs='+')
    
    args = parser.parse_args()
    train_model(**vars(args))


if __name__ == '__main__':
    main()