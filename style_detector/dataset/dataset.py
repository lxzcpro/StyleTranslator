import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple
import tempfile
import os


class StyleDataset(Dataset):
    """Dataset for style detection from CSV files"""
    
    def __init__(
        self,
        csv_path: str,
        tokenizer_name: str,
        max_length: int = 512,
        language_filter: str = None
    ):
        """
        Args:
            csv_path: Path to CSV file with columns: text, language, style
            tokenizer_name: Name of the pretrained tokenizer
            max_length: Maximum sequence length
            language_filter: Filter data by language ('chinese' or 'english')
        """
        self.data = pd.read_csv(csv_path)
        
        # Filter by language if specified
        if language_filter == 'chinese':
            self.data = self.data[self.data['language'] == 'ch'].reset_index(drop=True)
        elif language_filter == 'english':
            self.data = self.data[self.data['language'] == 'en'].reset_index(drop=True)

        # Validate that we have data after filtering
        if len(self.data) == 0:
            raise ValueError(f"No data found for language filter: {language_filter}. Check your CSV file and language codes.")

        # Create label encoder
        self.unique_styles = sorted(self.data['style'].unique())
        self.style_to_id = {style: idx for idx, style in enumerate(self.unique_styles)}
        self.id_to_style = {idx: style for style, idx in self.style_to_id.items()}
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        text = str(row['text'])
        style = row['style']
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.style_to_id[style], dtype=torch.long)
        }
    
    @property
    def num_classes(self) -> int:
        return len(self.unique_styles)
    
    def get_style_labels(self) -> List[str]:
        return self.unique_styles


def create_data_splits(
    csv_path: str,
    tokenizer_name: str,
    language_filter: str = None,
    max_length: int = 512,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[StyleDataset, StyleDataset, StyleDataset]:
    """
    Create train, validation, and test datasets from a CSV file
    
    Args:
        csv_path: Path to CSV file
        tokenizer_name: Name of pretrained tokenizer
        language_filter: Language to filter ('chinese' or 'english')
        max_length: Maximum sequence length
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Load data
    data = pd.read_csv(csv_path)
    
    if language_filter == 'chinese':
        data = data[data['language'] == 'ch'].reset_index(drop=True)
    elif language_filter == 'english':
        data = data[data['language'] == 'en'].reset_index(drop=True)

    # Validate that we have data after filtering
    if len(data) == 0:
        raise ValueError(f"No data found for language filter: {language_filter}. Check your CSV file and language codes.")

    # Shuffle data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split data
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    # Create temporary files with unique names to avoid race conditions
    # Use delete=False so files persist long enough for StyleDataset to read them
    train_temp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    val_temp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    test_temp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)

    try:
        # Save split data to temporary CSV files
        train_data.to_csv(train_temp.name, index=False)
        val_data.to_csv(val_temp.name, index=False)
        test_data.to_csv(test_temp.name, index=False)

        # Close the file handles
        train_temp.close()
        val_temp.close()
        test_temp.close()

        # Create datasets (they will read and store the data)
        train_dataset = StyleDataset(train_temp.name, tokenizer_name, max_length)
        val_dataset = StyleDataset(val_temp.name, tokenizer_name, max_length)
        test_dataset = StyleDataset(test_temp.name, tokenizer_name, max_length)

        return train_dataset, val_dataset, test_dataset
    finally:
        # Clean up temporary files
        for temp_file in [train_temp, val_temp, test_temp]:
            try:
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
            except Exception as e:
                # Log but don't fail if cleanup fails
                pass