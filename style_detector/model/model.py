import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig
from torchmetrics import Accuracy, F1Score, Precision, Recall
from typing import Dict, Any, Optional
import math


class StyleDetector(pl.LightningModule):
    """BERT-based style detector"""
    
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        dropout_rate: float = 0.1,
        freeze_bert_layers: int = 0,
        total_training_steps: int = None
    ):
        """
        Args:
            model_name: Pretrained BERT model name
            num_classes: Number of style classes
            learning_rate: Learning rate for optimization
            warmup_steps: Number of warmup steps for scheduler
            dropout_rate: Dropout rate for classification head
            freeze_bert_layers: Number of BERT layers to freeze (0 = no freezing)
            total_training_steps: Total number of training steps (for scheduler)
        """
        super().__init__()
        self.save_hyperparameters()

        # Validate freeze_bert_layers parameter
        if freeze_bert_layers < 0:
            raise ValueError(f"freeze_bert_layers must be >= 0, got {freeze_bert_layers}")

        # Load pretrained BERT
        try:
            self.config = AutoConfig.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model '{model_name}': {e}") from e

        # Validate and freeze layers if specified
        num_bert_layers = len(self.bert.encoder.layer)
        if freeze_bert_layers > num_bert_layers:
            print(f"Warning: freeze_bert_layers={freeze_bert_layers} exceeds available layers={num_bert_layers}. "
                  f"Freezing all {num_bert_layers} layers.")
            freeze_bert_layers = num_bert_layers
            self.hparams.freeze_bert_layers = freeze_bert_layers

        if freeze_bert_layers > 0:
            self._freeze_bert_layers(freeze_bert_layers)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics (top_k is not needed for multiclass classification)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average='macro')

        # Reusable test metrics to avoid recreating them each epoch
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

        self.test_step_outputs = []
        
    def _freeze_bert_layers(self, num_layers: int):
        """Freeze the first num_layers of BERT"""
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze encoder layers
        for i in range(min(num_layers, len(self.bert.encoder.layer))):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use pooled output (CLS token representation)
        # pooled_output = outputs.pooler_output
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Forward pass
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, labels)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Forward pass
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=False)
        self.log('val_precision', self.val_precision, prog_bar=False)
        self.log('val_recall', self.val_recall, prog_bar=False)
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Forward pass
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Calculate predictions
        preds = torch.argmax(logits, dim=1)
        
        # Store outputs manually
        output = {'loss': loss, 'preds': preds, 'labels': labels}
        self.test_step_outputs.append(output)
        
        return output
    
    def on_test_epoch_end(self):  # Removed 'outputs' argument
        """Calculate test metrics at epoch end"""
        # Retrieve stored outputs
        outputs = self.test_step_outputs

        if not outputs:
            return

        # Gather all predictions and labels
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_labels = torch.cat([x['labels'] for x in outputs])
        all_losses = torch.stack([x['loss'] for x in outputs])

        # Calculate metrics using reusable metric objects
        test_loss = all_losses.mean()
        test_acc = self.test_accuracy(all_preds, all_labels)
        test_f1 = self.test_f1(all_preds, all_labels)

        # Log metrics
        self.log('test_loss', test_loss)
        self.log('test_acc', test_acc)
        self.log('test_f1', test_f1)

        print(f"\nTest Results:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"F1 Score: {test_f1:.4f}")

        # Clear the list to free memory
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers with proper warmup and decay"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01
        )

        # Improved warmup scheduler with cosine decay after warmup
        def lr_lambda(current_step):
            warmup_steps = self.hparams.warmup_steps

            # Linear warmup
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            # Cosine decay after warmup
            if self.hparams.total_training_steps is not None:
                progress = float(current_step - warmup_steps) / float(
                    max(1, self.hparams.total_training_steps - warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(progress * math.pi)))

            # If total_training_steps not provided, use linear decay
            # Decay to 0.1 of initial LR over 10x the warmup steps
            decay_steps = max(warmup_steps * 10, 1000)
            progress = min(1.0, float(current_step - warmup_steps) / float(decay_steps))
            return max(0.1, 1.0 - 0.9 * progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def predict_style(self, text: str, tokenizer) -> Dict[str, Any]:
        """
        Predict style for a single text

        Args:
            text: Input text
            tokenizer: Tokenizer instance

        Returns:
            Dictionary with prediction and probabilities

        Raises:
            ValueError: If text is empty or None
            RuntimeError: If prediction fails
        """
        # Validate input
        if not isinstance(text, str) or text.strip() == "":
            raise ValueError("Text must be a non-empty string")

        try:
            self.eval()

            # Tokenize
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )

            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # Predict
            with torch.no_grad():
                logits = self(input_ids, attention_mask)
                probs = F.softmax(logits, dim=-1)
                pred_class = torch.argmax(logits, dim=-1)

            return {
                'predicted_class': pred_class.item(),
                'probabilities': probs.squeeze().cpu().numpy()
            }
        except (ValueError, TypeError):
            raise
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}") from e