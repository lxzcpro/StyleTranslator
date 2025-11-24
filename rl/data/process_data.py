import os
import argparse
from datasets import Dataset, load_dataset
from tqdm import tqdm
import json
import random
import pandas as pd
from transformers import AutoTokenizer

# Language code to language name mapping table
language_map = {
    'en': 'English',
    'de': 'German',
    'zh': 'Chinese',
    'ja': 'Japanese',
}

def make_prefix(example, template_type, tokenizer):
    """
    Dynamically generate prompt text
    """
    lg = example.get('lg', '')
    source_lang, target_lang = lg.split('-') if '-' in lg else ('unknown', 'unknown')

    src_lang_name = language_map.get(source_lang, source_lang.capitalize())
    tgt_lang_name = language_map.get(target_lang, target_lang.capitalize())

    user_input = example.get("src_text", "")
    solution = example.get("tgt_text", "")

    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The User asks for a translation from {src_lang_name} to {tgt_lang_name}, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the final translation. The reasoning process and final translation are enclosed within <think> </think> and <translate> </translate> tags, respectively, i.e., <think> reasoning process here <translate> final translation here </translate>. \n\nUser:{user_input}\nAssistant:"""
    elif template_type == 'chat':
        messages = [
        {"role": "system", "content": f"You are a helpful translation assistant. There is a conversation between User and Assistant. The user asks for a translation from {src_lang_name} to {tgt_lang_name}, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the final translation. The reasoning process and final translation are enclosed within <think> </think> and <translate> </translate> tags, respectively, i.e., <think> reasoning process here </think><translate> final translation here </translate>."},
        {"role": "user", "content": user_input}
                ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prefix = text
    elif template_type == 'rl':
        prefix = f"""A conversation between User and Assistant. The User asks for a translation from {src_lang_name} to {tgt_lang_name}, and the Assistant translates it. The final translation are enclosed within <translate> </translate> tags, i.e., <translate> final translation here </translate>. \n\nUser:{user_input}\nAssistant:"""    
    
    return prefix

def preprocess_data(data):
    """
    Preprocess data to ensure each sample only contains necessary fields:
    - data_source
    - lang_pair
    - src_text
    - tgt_text
    """
    processed_data = []
    for example in data:
        lg = example.get('lg', 'en-zh')  # Get language pair
        source_lang, target_lang = lg.split('-')  # Split language pair

        # Dynamically extract source language and target language text
        src_text = example.get(source_lang, "")
        tgt_text = example.get(target_lang, "")

        # Construct new sample
        processed_example = {
            'data_source': example.get('data_source', 'unknown'),
            'lg': lg,
            'src_text': src_text,
            'tgt_text': tgt_text
        }

        processed_data.append(processed_example)
    return processed_data

def extract_data(example):
    """
    Extract fields from example data (assuming example is already an expanded dictionary)
    """
    data_source = example.get('data_source', 'unknown')
    lg = example.get('lg', 'en-zh')  # Ensure lg field exists
    source_lang, target_lang = lg.split('-')
    
    # Directly get corresponding language fields
    source = example.get(source_lang, "")
    solution = example.get(target_lang, "")
    
    return {
        'data_source': data_source,
        'lang_pair': lg,
        'src_text': source,
        'tgt_text': solution,
    }

def read_jsonl_files(file_paths):
    data = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Directly parse as dictionary, no nesting needed
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[Line {line_num}] JSON parse failed → Line content: {repr(line)}")
    return data

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prepare translation dataset')
    parser.add_argument('--train_files', nargs='+', default=['train/train_zhen_6565.jsonl', 'train/train_enzh_6565.jsonl'], help='Training JSONL files')
    parser.add_argument('--test_files', nargs='+', default=['test/wmt23_zhen.jsonl', 'test/wmt24_enzh.jsonl'], help='Test JSONL files')
    parser.add_argument('--tokenizer_path', type=str, default='../Qwen2.5-3B-Instruct', help='Path to the tokenizer')
    parser.add_argument('--template_type', type=str, choices=['base', 'chat', 'rl'], default='chat', help='Template type for prompts')
    parser.add_argument('--train_sample_size', type=int, default=10000000, help='Number of training samples to use')
    parser.add_argument('--test_sample_size', type=int, default=10000000, help='Number of test samples to use')
    parser.add_argument('--train_output_file', type=str, default='train_qwen_chat.parquet', help='Output filename for train data')
    parser.add_argument('--test_output_file', type=str, default='test_qwen_chat.parquet', help='Output filename for test data')
    
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # Read training data
    train_data = preprocess_data(read_jsonl_files(args.train_files))
    train_dataset = Dataset.from_list(train_data)

    # Read test data
    test_data = preprocess_data(read_jsonl_files(args.test_files))
    test_dataset = Dataset.from_list(test_data)

    def make_map_fn(split):
        def process_fn(example, idx):
            # Dynamic data extraction
            extracted_data = extract_data(example)
            lg = extracted_data['lang_pair']
            source_lang, target_lang = lg.split('-')
            
            # Dynamic source and target language field extraction
            source = example['src_text']
            solution = example['tgt_text']

            # Generate prefix
            question = make_prefix(example, template_type=args.template_type, tokenizer=tokenizer)
            
            data = {
                "data_source": extracted_data['data_source'] + "_" + lg,
                "lang_pair": lg,
                "src_text": source,
                "tgt_text": solution,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "translate",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # Sampling
    if len(train_dataset) > args.train_sample_size:
        train_dataset = train_dataset.shuffle()
        train_dataset = train_dataset.select(range(args.train_sample_size))

    if len(test_dataset) > args.test_sample_size:
        test_dataset = test_dataset.shuffle()
        test_dataset = test_dataset.select(range(args.test_sample_size))

    # Save datasets
    train_output_path = os.path.join(args.train_output_file)
    test_output_path = os.path.join(args.test_output_file)
    
    train_dataset.to_parquet(train_output_path)
    test_dataset.to_parquet(test_output_path)

    # Print dataset format
    print("Parquet dataset format:")

    print("Train dataset columns:")
    train_pdf = train_dataset.to_pandas()
    print(train_pdf.head())
    print(train_pdf['prompt'][0])

    print("\nTest dataset columns:")
    test_pdf = test_dataset.to_pandas()
    print(test_pdf.head())
    
    print(f"Train dataset saved to: {train_output_path}")
    print(f"Test dataset saved to: {test_output_path}")

if __name__ == '__main__':
    main()

