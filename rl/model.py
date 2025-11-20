import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig
from torchmetrics import Accuracy, F1Score, Precision, Recall
from typing import Dict, Any, Optional


class StyleDetector(pl.LightningModule):
    """BERT-based style detector"""
    
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        dropout_rate: float = 0.1,
        freeze_bert_layers: int = 0
    ):
        """
        Args:
            model_name: Pretrained BERT model name
            num_classes: Number of style classes
            learning_rate: Learning rate for optimization
            warmup_steps: Number of warmup steps for scheduler
            dropout_rate: Dropout rate for classification head
            freeze_bert_layers: Number of BERT layers to freeze (0 = no freezing)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Load pretrained BERT
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze layers if specified
        if freeze_bert_layers > 0:
            self._freeze_bert_layers(freeze_bert_layers)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average='macro')

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
        
        # Calculate metrics
        test_loss = all_losses.mean()
        test_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes).to(self.device)(all_preds, all_labels)
        test_f1 = F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='macro').to(self.device)(all_preds, all_labels)
        
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
        """Configure optimizers and schedulers"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01
        )
        
        # Linear warmup scheduler
        def lr_lambda(current_step):
            if current_step < self.hparams.warmup_steps:
                return float(current_step) / float(max(1, self.hparams.warmup_steps))
            return 1.0
        
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
        """
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