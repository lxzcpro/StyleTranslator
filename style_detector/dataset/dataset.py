import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple
import tempfile
import os
import warnings


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
        # Validate CSV file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load CSV
        try:
            self.data = pd.read_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV file '{csv_path}': {e}") from e

        # Validate required columns
        required_columns = ['text', 'language', 'style']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"CSV file is missing required columns: {missing_columns}. "
                           f"Required columns: {required_columns}, Found: {list(self.data.columns)}")
        
        # Clean data before filtering
        original_len = len(self.data)

        # Handle missing values in text column
        null_text_count = self.data['text'].isnull().sum()
        if null_text_count > 0:
            warnings.warn(f"Found {null_text_count} rows with null/NaN text values. These will be removed.")
            self.data = self.data.dropna(subset=['text']).reset_index(drop=True)

        # Convert text to string and remove empty/whitespace-only texts
        self.data['text'] = self.data['text'].astype(str)
        empty_text_mask = self.data['text'].str.strip() == ''
        empty_text_count = empty_text_mask.sum()
        if empty_text_count > 0:
            warnings.warn(f"Found {empty_text_count} rows with empty/whitespace-only text. These will be removed.")
            self.data = self.data[~empty_text_mask].reset_index(drop=True)

        # Handle missing values in style column
        null_style_count = self.data['style'].isnull().sum()
        if null_style_count > 0:
            warnings.warn(f"Found {null_style_count} rows with null/NaN style labels. These will be removed.")
            self.data = self.data.dropna(subset=['style']).reset_index(drop=True)

        cleaned_count = original_len - len(self.data)
        if cleaned_count > 0:
            print(f"Data cleaning: Removed {cleaned_count} invalid rows ({cleaned_count/original_len*100:.1f}%)")

        # Filter by language if specified
        if language_filter == 'chinese':
            self.data = self.data[self.data['language'] == 'ch'].reset_index(drop=True)
        elif language_filter == 'english':
            self.data = self.data[self.data['language'] == 'en'].reset_index(drop=True)

        # Validate that we have data after filtering
        if len(self.data) == 0:
            raise ValueError(f"No data found for language filter: {language_filter}. Check your CSV file and language codes.")

        # Remove duplicates
        duplicate_count = self.data.duplicated(subset=['text']).sum()
        if duplicate_count > 0:
            warnings.warn(f"Found {duplicate_count} duplicate texts. Keeping first occurrence.")
            self.data = self.data.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)

        # Create label encoder
        self.unique_styles = sorted(self.data['style'].unique())
        self.style_to_id = {style: idx for idx, style in enumerate(self.unique_styles)}
        self.id_to_style = {idx: style for style, idx in self.style_to_id.items()}

        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer '{tokenizer_name}': {e}") from e

        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            row = self.data.iloc[idx]
            text = str(row['text'])
            style = row['style']

            # Additional validation at runtime
            if not text or text.strip() == '':
                raise ValueError(f"Empty text found at index {idx}")

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
        except Exception as e:
            raise RuntimeError(f"Error processing item at index {idx}: {e}") from e
    
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
    val_ratio: float = 0.1,
    stratify: bool = True,
    random_state: int = 42
) -> Tuple[StyleDataset, StyleDataset, StyleDataset]:
    """
    Create train, validation, and test datasets from a CSV file with stratified splitting

    Args:
        csv_path: Path to CSV file
        tokenizer_name: Name of pretrained tokenizer
        language_filter: Language to filter ('chinese' or 'english')
        max_length: Maximum sequence length
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        stratify: Whether to use stratified splitting to maintain class balance
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Validate input
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0")

    # Load data
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV file '{csv_path}': {e}") from e

    # Validate required columns
    required_columns = ['text', 'language', 'style']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"CSV file is missing required columns: {missing_columns}")

    if language_filter == 'chinese':
        data = data[data['language'] == 'ch'].reset_index(drop=True)
    elif language_filter == 'english':
        data = data[data['language'] == 'en'].reset_index(drop=True)

    # Validate that we have data after filtering
    if len(data) == 0:
        raise ValueError(f"No data found for language filter: {language_filter}. Check your CSV file and language codes.")

    # Stratified or random splitting
    if stratify:
        # Use sklearn's train_test_split for stratified splitting
        try:
            from sklearn.model_selection import train_test_split

            # First split: train vs (val + test)
            train_data, temp_data = train_test_split(
                data,
                train_size=train_ratio,
                random_state=random_state,
                stratify=data['style']
            )

            # Second split: val vs test
            val_ratio_adjusted = val_ratio / (1 - train_ratio)
            val_data, test_data = train_test_split(
                temp_data,
                train_size=val_ratio_adjusted,
                random_state=random_state,
                stratify=temp_data['style']
            )

            print(f"Stratified split created: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

            # Print class distribution
            print("Class distribution:")
            for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
                dist = split_data['style'].value_counts(normalize=True).sort_index()
                print(f"  {split_name}: {dict(dist)}")

        except ImportError:
            warnings.warn("scikit-learn not installed. Falling back to random splitting.")
            stratify = False

    if not stratify:
        # Random splitting
        data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Calculate split indices
        n_total = len(data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Split data
        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]

    # Create temporary files with unique names to avoid race conditions
    temp_files = []
    train_temp = None
    val_temp = None
    test_temp = None

    try:
        # Create temporary files (delete=False so they persist)
        train_temp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, prefix='train_')
        val_temp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, prefix='val_')
        test_temp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, prefix='test_')
        temp_files = [train_temp, val_temp, test_temp]

        # Save split data to temporary CSV files
        train_data.to_csv(train_temp.name, index=False)
        val_data.to_csv(val_temp.name, index=False)
        test_data.to_csv(test_temp.name, index=False)

        # Close the file handles
        train_temp.close()
        val_temp.close()
        test_temp.close()

        # Create datasets (they will read and store the data internally)
        train_dataset = StyleDataset(train_temp.name, tokenizer_name, max_length)
        val_dataset = StyleDataset(val_temp.name, tokenizer_name, max_length)
        test_dataset = StyleDataset(test_temp.name, tokenizer_name, max_length)

        return train_dataset, val_dataset, test_dataset

    except Exception as e:
        raise RuntimeError(f"Failed to create data splits: {e}") from e

    finally:
        # Clean up temporary files robustly
        for temp_file in temp_files:
            if temp_file is not None:
                try:
                    # Ensure file is closed
                    if not temp_file.closed:
                        temp_file.close()
                    # Remove the file
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                except Exception as e:
                    # Log warning but don't fail
                    warnings.warn(f"Failed to clean up temporary file {temp_file.name}: {e}")