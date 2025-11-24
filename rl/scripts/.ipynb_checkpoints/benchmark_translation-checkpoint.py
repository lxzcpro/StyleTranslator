#!/usr/bin/env python3
"""
Benchmark script for evaluating translation models.
Features:
- Dynamic Source/Target Language Detection
- Gated Style Scoring (Style=0 if Language Compliance Fails)
- Detailed BLEU Breakdown (1-4)
- Individual COMET Scores
"""

import argparse
import logging
import sys
import os
import warnings
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import time

# Suppress noisy warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="transformers")

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rewards import RewardFactory
from src.rewards.format import FormatReward

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""
    # BLEU scores (Corpus Level)
    bleu_score: float = 0.0
    bleu_1: float = 0.0
    bleu_2: float = 0.0
    bleu_3: float = 0.0
    bleu_4: float = 0.0

    # COMET score (Average)
    comet_score: float = 0.0

    # Style score
    style_score: float = 0.0
    style_std: float = 0.0

    # Format compliance
    format_score: float = 0.0
    format_compliance_rate: float = 0.0

    # Language compliance
    language_compliance_rate: float = 0.0

    # Aggregate
    total_reward: float = 0.0

    # Meta
    num_samples: int = 0
    generation_time: float = 0.0
    tokens_per_second: float = 0.0


@dataclass
class SampleResult:
    """Result for a single sample."""
    source: str
    reference: str
    hypothesis: str
    bleu: float       # Main BLEU score
    bleu_1: float     # Unigram precision
    bleu_2: float     # Bigram precision
    bleu_3: float     # Trigram precision
    bleu_4: float     # 4-gram precision
    comet: float
    style: float
    format_score: float
    is_valid: bool
    lang_valid: bool
    detected_src_lang: str
    expected_tgt_lang: str


class TranslationBenchmark:
    """Benchmark translation models with standard metrics."""

    def __init__(
        self,
        base_model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        rl_model_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "auto",
        test_mode: bool = False
    ):
        self.base_model_path = base_model_path
        self.rl_model_path = rl_model_path
        self.test_mode = test_mode

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Resolve dtype
        if dtype == "auto":
            if self.device == "cuda":
                self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                self.dtype = torch.float32
        else:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32
            }
            self.dtype = dtype_map.get(dtype, torch.float32)

        logger.info(f"Device: {self.device}, dtype: {self.dtype}")

        self.tokenizer = None
        self.base_model = None
        self.rl_model = None
        self.format_reward = FormatReward()
        self.comet_model = None
        self.bleu_scorer = None

    def setup(
        self,
        chinese_bert_path: Optional[str] = None,
        english_bert_path: Optional[str] = None,
        load_base: bool = True,
        load_rl: bool = True
    ):
        self._setup_tokenizer()
        self._setup_bleu()
        self._setup_comet()
        self._setup_style_reward(chinese_bert_path, english_bert_path)

        if load_base:
            self._setup_base_model()
        if load_rl and self.rl_model_path:
            self._setup_rl_model()

    def _setup_tokenizer(self):
        logger.info(f"Loading tokenizer from {self.base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _setup_bleu(self):
        try:
            import sacrebleu
            self.bleu_scorer = sacrebleu
            logger.info("BLEU scorer (sacrebleu) initialized.")
        except ImportError:
            logger.warning("sacrebleu not installed. BLEU scores will be computed with nltk.")
            try:
                from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
                self.bleu_scorer = "nltk"
                logger.info("BLEU scorer (nltk) initialized.")
            except ImportError:
                logger.warning("nltk not installed. BLEU scores will be simulated.")
                self.bleu_scorer = None

    def _setup_comet(self):
        try:
            from comet import download_model, load_from_checkpoint
            # Use specific model ID
            model_path = download_model("Unbabel/wmt22-cometkiwi-da")
            self.comet_model = load_from_checkpoint(model_path)
            if self.device == "cuda":
                self.comet_model = self.comet_model.to(self.device)
            logger.info("COMET model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load COMET model: {e}. Using fallback scoring.")
            self.comet_model = None

    def _setup_style_reward(
        self,
        chinese_bert_path: Optional[str] = None,
        english_bert_path: Optional[str] = None
    ):
        config = {
            'reward': {
                'test_mode': self.test_mode,
                'chinese_bert_path': chinese_bert_path or '',
                'english_bert_path': english_bert_path or '',
                'style_types': ['law', 'literature', 'news', 'science'],
                'comet_device': self.device,
                'comet_model': "Unbabel/wmt22-cometkiwi-da"
            },
            'model': {
                'device': self.device
            }
        }
        self.reward_manager = RewardFactory.create_from_config(config)
        logger.info("Style reward model initialized.")

    def _setup_base_model(self):
        logger.info(f"Loading base model from {self.base_model_path}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=self.dtype,
            device_map=self.device if self.device == "auto" else None,
            trust_remote_code=True
        )
        if self.device != "auto":
            self.base_model = self.base_model.to(self.device)
        self.base_model.eval()
        logger.info("Base model loaded.")

    def _setup_rl_model(self):
        if not self.rl_model_path or not os.path.exists(self.rl_model_path):
            logger.warning(f"RL model path not found: {self.rl_model_path}")
            return
        logger.info(f"Loading RL model from {self.rl_model_path}...")
        rl_base = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=self.dtype,
            device_map=self.device if self.device == "auto" else None,
            trust_remote_code=True
        )
        self.rl_model = PeftModel.from_pretrained(rl_base, self.rl_model_path)
        if self.device != "auto":
            self.rl_model = self.rl_model.to(self.device)
        self.rl_model.eval()
        logger.info("RL model loaded.")

    def format_prompt(self, source_text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a professional translator. Translate the text into Chinese. Output the translation enclosed in <translate> and </translate> tags."
            },
            {
                "role": "user",
                "content": "The weather is nice today."
            },
            {
                "role": "assistant",
                "content": "<translate>今天天气很好。</translate>"
            },
            {
                "role": "user",
                "content": source_text
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt += "<translate>"
        return prompt

    def generate(
        self,
        model,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return "<translate>" + generated_text

    def detect_language(self, text: str) -> str:
        """
        Detect if text is predominantly Chinese ('zh') or English ('en').
        Returns 'unknown' if unclear.
        """
        if not text or len(text.strip()) == 0:
            return "unknown"
        
        clean_text = re.sub(r'\s|\d|[.,!?;:()"\'-]', '', text)
        if not clean_text:
            return "unknown"

        total_len = len(text)
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))

        zh_ratio = chinese_chars / total_len
        en_ratio = english_chars / total_len

        if zh_ratio > 0.2:
            return "zh"
        if en_ratio > 0.2:
            return "en"
        return "unknown"

    def check_language_compliance(self, hypothesis: str, expected_lang: str) -> bool:
        """Check if hypothesis matches the expected language."""
        detected = self.detect_language(hypothesis)
        if expected_lang == "zh":
            return detected == "zh"
        if expected_lang == "en":
            return detected == "en"
        return False

    def compute_sentence_bleu(self, hypothesis: str, reference: str) -> Dict[str, float]:
        """Compute sentence-level BLEU scores breakdown."""
        default_scores = {'bleu': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
        
        if not hypothesis or not reference:
            return default_scores
        
        # Determine tokenization based on reference language
        ref_lang = self.detect_language(reference)
        tokenize_mode = 'zh' if ref_lang == 'zh' else '13a'

        if hasattr(self.bleu_scorer, 'sentence_bleu'):
            try:
                res = self.bleu_scorer.sentence_bleu(hypothesis, [reference], tokenize=tokenize_mode)
                return {
                    'bleu': res.score,
                    'bleu_1': res.precisions[0] if len(res.precisions) > 0 else 0.0,
                    'bleu_2': res.precisions[1] if len(res.precisions) > 1 else 0.0,
                    'bleu_3': res.precisions[2] if len(res.precisions) > 2 else 0.0,
                    'bleu_4': res.precisions[3] if len(res.precisions) > 3 else 0.0,
                }
            except Exception:
                return default_scores
        elif self.bleu_scorer == "nltk":
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            smoother = SmoothingFunction().method1
            ref_tokens = [list(reference)]
            hyp_tokens = list(hypothesis)
            
            return {
                'bleu': sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoother) * 100,
                'bleu_1': sentence_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoother) * 100,
                'bleu_2': sentence_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother) * 100,
                'bleu_3': sentence_bleu(ref_tokens, hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoother) * 100,
                'bleu_4': sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother) * 100,
            }
        else:
            # Fallback
            overlap = len(set(hypothesis) & set(reference))
            score = (overlap / max(len(set(hypothesis)), len(set(reference)), 1)) * 100
            return {'bleu': score, 'bleu_1': score, 'bleu_2': score, 'bleu_3': score, 'bleu_4': score}

    def compute_corpus_bleu(self, hypotheses: List[str], references: List[str]) -> Dict[str, float]:
        if not hypotheses or not references:
            return {'bleu': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}

        # Heuristic: check first reference to decide tokenizer for corpus
        first_ref_lang = self.detect_language(references[0]) if references else 'en'
        tokenize_mode = 'zh' if first_ref_lang == 'zh' else '13a'

        if hasattr(self.bleu_scorer, 'corpus_bleu'):
            try:
                bleu = self.bleu_scorer.corpus_bleu(hypotheses, [references], tokenize=tokenize_mode)
                return {
                    'bleu': bleu.score,
                    'bleu_1': bleu.precisions[0] if len(bleu.precisions) > 0 else 0.0,
                    'bleu_2': bleu.precisions[1] if len(bleu.precisions) > 1 else 0.0,
                    'bleu_3': bleu.precisions[2] if len(bleu.precisions) > 2 else 0.0,
                    'bleu_4': bleu.precisions[3] if len(bleu.precisions) > 3 else 0.0,
                }
            except Exception as e:
                logger.warning(f"sacrebleu error: {e}")
                return {'bleu': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
        
        sentence_scores = [self.compute_sentence_bleu(h, r)['bleu'] for h, r in zip(hypotheses, references)]
        mean_score = np.mean(sentence_scores) if sentence_scores else 0.0
        return {
            'bleu': mean_score, 'bleu_1': mean_score, 'bleu_2': mean_score, 
            'bleu_3': mean_score, 'bleu_4': mean_score,
        }

    def compute_comet(self, sources: List[str], hypotheses: List[str], references: List[str]) -> Tuple[float, List[float]]:
        if self.comet_model is None:
            scores = []
            for src, hyp, ref in zip(sources, hypotheses, references):
                if not hyp or not ref:
                    scores.append(0.0)
                    continue
                overlap = len(set(hyp) & set(ref))
                score = overlap / max(len(set(hyp)), len(set(ref)), 1)
                scores.append(score)
            return float(np.mean(scores)), scores

        try:
            data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(sources, hypotheses, references)]
            output = self.comet_model.predict(data, batch_size=8, gpus=1 if self.device == "cuda" else 0)
            return float(output.system_score), output.scores
        except Exception as e:
            logger.warning(f"COMET error: {e}")
            return 0.0, [0.0] * len(sources)

    def compute_style_score(self, source: str, hypothesis: str, lang_pair: str = "en-zh") -> float:
        try:
            result = self.reward_manager.calculate_single_reward(
                generated_text=f"<translate>{hypothesis}</translate>",
                source_text=source,
                prompt="",
                language_pair=lang_pair,
                reference_text=source
            )
            return result.components.style_score
        except Exception as e:
            logger.debug(f"Style score error: {e}")
            return 0.0

    def run_benchmark(
        self,
        model,
        model_name: str,
        test_data: List[Dict[str, Any]],
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> tuple[BenchmarkMetrics, List[SampleResult]]:
        """Run full benchmark on a model with Gated Scoring."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {model_name}")
        logger.info(f"Samples: {len(test_data)}")
        logger.info("Mode: Cross-lingual validation with Gated Scoring")
        logger.info(f"{'='*60}")

        sources, references, hypotheses = [], [], []
        sample_results = []
        format_scores, style_scores = [], []
        valid_count, lang_valid_count = 0, 0
        total_tokens = 0

        start_time = time.time()

        for item in tqdm(test_data, desc=f"Generating ({model_name})"):
            source = item['src_text']
            reference = item.get('tgt_text', '')
            dataset_lang_pair = item.get('lang_pair', 'en-zh')

            # --- 1. DYNAMIC LANGUAGE DETECTION ---
            detected_src_lang = self.detect_language(source)
            
            # Determine Expected Target Language (Opposite of Source)
            if detected_src_lang == "zh":
                expected_tgt_lang = "en"
                dynamic_lang_pair = "zh-en"
            elif detected_src_lang == "en":
                expected_tgt_lang = "zh"
                dynamic_lang_pair = "en-zh"
            else:
                dynamic_lang_pair = dataset_lang_pair
                expected_tgt_lang = dataset_lang_pair.split('-')[-1] if '-' in dataset_lang_pair else 'zh'

            # Generate
            prompt = self.format_prompt(source)
            generated = self.generate(model, prompt, max_new_tokens, temperature)

            # Extract translation
            translation = self.format_reward.extract_translation(generated)
            if not translation:
                translation = generated.replace("<translate>", "").replace("</translate>", "").strip()

            # --- 2. VALIDATION CHECKS ---
            format_result = self.format_reward.calculate(generated_text=generated, prompt=prompt)
            format_score = format_result.score
            is_valid = format_score > 0.3
            
            is_lang_valid = self.check_language_compliance(translation, expected_tgt_lang)

            # --- 3. SCORING ---
            # Gated Style Scoring
            if is_lang_valid:
                style_score = self.compute_style_score(source, translation, dynamic_lang_pair)
            else:
                style_score = 0.0
            
            # Sentence BLEU Breakdown
            bleu_results = self.compute_sentence_bleu(translation, reference)

            # Update Lists
            sources.append(source)
            references.append(reference)
            hypotheses.append(translation)
            format_scores.append(format_score)
            style_scores.append(style_score)

            if is_valid: valid_count += 1
            if is_lang_valid: lang_valid_count += 1
            total_tokens += len(self.tokenizer.encode(translation))

            sample_results.append(SampleResult(
                source=source,
                reference=reference,
                hypothesis=translation,
                bleu=bleu_results['bleu'],
                bleu_1=bleu_results['bleu_1'],
                bleu_2=bleu_results['bleu_2'],
                bleu_3=bleu_results['bleu_3'],
                bleu_4=bleu_results['bleu_4'],
                comet=0.0, # Filled later
                style=style_score,
                format_score=format_score,
                is_valid=is_valid,
                lang_valid=is_lang_valid,
                detected_src_lang=detected_src_lang,
                expected_tgt_lang=expected_tgt_lang
            ))

        generation_time = time.time() - start_time

        logger.info("Computing Corpus BLEU scores...")
        bleu_scores = self.compute_corpus_bleu(hypotheses, references)
        
        logger.info("Computing COMET scores...")
        comet_score, comet_individual_scores = self.compute_comet(sources, hypotheses, references)

        # Fill back individual COMET scores
        for i, res in enumerate(sample_results):
            if i < len(comet_individual_scores):
                res.comet = comet_individual_scores[i]

        metrics = BenchmarkMetrics(
            bleu_score=bleu_scores['bleu'],
            bleu_1=bleu_scores['bleu_1'],
            bleu_2=bleu_scores['bleu_2'],
            bleu_3=bleu_scores['bleu_3'],
            bleu_4=bleu_scores['bleu_4'],
            comet_score=comet_score,
            style_score=float(np.mean(style_scores)),
            style_std=float(np.std(style_scores)),
            format_score=float(np.mean(format_scores)),
            format_compliance_rate=valid_count / len(test_data) if test_data else 0.0,
            language_compliance_rate=lang_valid_count / len(test_data) if test_data else 0.0,
            total_reward=float(np.mean(format_scores)) + comet_score + float(np.mean(style_scores)),
            num_samples=len(test_data),
            generation_time=generation_time,
            tokens_per_second=total_tokens / generation_time if generation_time > 0 else 0.0
        )

        return metrics, sample_results

    def print_results(self, base_metrics: Optional[BenchmarkMetrics] = None, rl_metrics: Optional[BenchmarkMetrics] = None):
        print("\n" + "=" * 70)
        print("                    TRANSLATION BENCHMARK RESULTS")
        print("=" * 70)

        def print_metrics(name: str, m: BenchmarkMetrics):
            print(f"\n{name}")
            print("-" * 50)
            print(f"  Samples:              {m.num_samples}")
            print(f"  Generation Time:      {m.generation_time:.2f}s ({m.tokens_per_second:.1f} tok/s)")
            print()
            print(f"  BLEU Score:           {m.bleu_score:.2f}")
            print(f"    - BLEU-1:           {m.bleu_1:.2f}")
            print(f"    - BLEU-4:           {m.bleu_4:.2f}")
            print()
            print(f"  COMET Score:          {m.comet_score:.4f}")
            print(f"  Style Score:          {m.style_score:.4f} (+/- {m.style_std:.4f})")
            print(f"  Format Score:         {m.format_score:.4f}")
            print(f"  Format Compliance:    {m.format_compliance_rate*100:.1f}%")
            print(f"  Language Compliance:  {m.language_compliance_rate*100:.1f}%")
            print()
            print(f"  Total Reward:         {m.total_reward:.4f}")

        if base_metrics: print_metrics("Original Model", base_metrics)
        if rl_metrics: print_metrics("RL Fine-tuned Model", rl_metrics)

        if base_metrics and rl_metrics:
            print("\n" + "-" * 50)
            print("COMPARISON (RL - Original)")
            print("-" * 50)
            def delta(rl_val, base_val):
                diff = rl_val - base_val
                sign = "+" if diff >= 0 else ""
                return f"{sign}{diff:.4f}"
            print(f"  BLEU:                 {delta(rl_metrics.bleu_score, base_metrics.bleu_score)}")
            print(f"  COMET:                {delta(rl_metrics.comet_score, base_metrics.comet_score)}")
            print(f"  Style:                {delta(rl_metrics.style_score, base_metrics.style_score)}")
            print(f"  Language Compl.:      {delta(rl_metrics.language_compliance_rate*100, base_metrics.language_compliance_rate*100)}%")
            print(f"  Total Reward:         {delta(rl_metrics.total_reward, base_metrics.total_reward)}")
        print("\n" + "=" * 70)

    def print_samples(self, base_samples: Optional[List[SampleResult]] = None, rl_samples: Optional[List[SampleResult]] = None, num_samples: int = 5):
        print("\n" + "=" * 70)
        print("                      SAMPLE TRANSLATIONS")
        print("=" * 70)
        samples_to_show = min(num_samples, len(base_samples) if base_samples else len(rl_samples) if rl_samples else 0)

        for i in range(samples_to_show):
            print(f"\n--- Sample {i+1} ---")
            if base_samples:
                s = base_samples[i]
                print(f"Source ({s.detected_src_lang}): {s.source[:60]}...")
                print(f"[Original -> {s.expected_tgt_lang}] {s.hypothesis[:60]}...")
                print(f"            bleu={s.bleu:.2f}, lang_valid={s.lang_valid}")
            if rl_samples:
                s = rl_samples[i]
                if not base_samples:
                    print(f"Source ({s.detected_src_lang}): {s.source[:60]}...")
                print(f"[RL       -> {s.expected_tgt_lang}] {s.hypothesis[:60]}...")
                print(f"            bleu={s.bleu:.2f}, lang_valid={s.lang_valid}")
        print("\n" + "=" * 70)

    def save_results(self, output_dir: str, base_metrics, rl_metrics, base_samples, rl_samples):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "timestamp": timestamp,
            "config": {
                "base_model": self.base_model_path,
                "rl_model": self.rl_model_path,
                "device": self.device,
                "test_mode": self.test_mode
            }
        }
        if base_metrics: results["base_model_metrics"] = asdict(base_metrics)
        if rl_metrics: results["rl_model_metrics"] = asdict(rl_metrics)
        
        metrics_path = os.path.join(output_dir, f"benchmark_metrics_{timestamp}.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Metrics saved to: {metrics_path}")

        if base_samples:
            base_path = os.path.join(output_dir, f"base_samples_{timestamp}.json")
            with open(base_path, 'w', encoding='utf-8') as f:
                json.dump([asdict(s) for s in base_samples], f, ensure_ascii=False, indent=2)
        if rl_samples:
            rl_path = os.path.join(output_dir, f"rl_samples_{timestamp}.json")
            with open(rl_path, 'w', encoding='utf-8') as f:
                json.dump([asdict(s) for s in rl_samples], f, ensure_ascii=False, indent=2)

def load_test_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        path = Path(__file__).parent.parent.parent / file_path
        if not path.exists():
            raise FileNotFoundError(f"Test file not found: {file_path}")
    logger.info(f"Loading test data from {path}...")
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    elif path.suffix == '.jsonl':
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    
    col_map = {}
    for src_col in ['src_text', 'source', 'src']:
        if src_col in df.columns: col_map['src'] = src_col; break
    for tgt_col in ['tgt_text', 'target', 'tgt']:
        if tgt_col in df.columns: col_map['tgt'] = tgt_col; break
    if 'src' not in col_map:
        raise ValueError(f"No source column found. Available: {df.columns.tolist()}")
    if max_samples and len(df) > max_samples:
        df = df.head(max_samples)
    
    data = []
    for _, row in df.iterrows():
        data.append({
            'src_text': row[col_map['src']],
            'tgt_text': row.get(col_map.get('tgt', ''), ''),
            'lang_pair': row.get('lang_pair', 'en-zh')
        })
    logger.info(f"Loaded {len(data)} samples.")
    return data

def main():
    parser = argparse.ArgumentParser(description="Benchmark translation models")
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--rl_model_path", type=str, default=None)
    parser.add_argument("--test_file", type=str, default="rl/data/test/parquet/test_style.parquet")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--test_mode", action="store_true", help="Use mock models")
    parser.add_argument("--chinese_bert_path", type=str, default=None)
    parser.add_argument("--english_bert_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/benchmark")
    parser.add_argument("--show_samples", type=int, default=5)
    parser.add_argument("--skip_base", action="store_true")
    parser.add_argument("--save", action="store_true", help="Save results to files")

    args = parser.parse_args()

    benchmark = TranslationBenchmark(
        base_model_path=args.base_model_path,
        rl_model_path=args.rl_model_path,
        device=args.device,
        dtype=args.dtype,
        test_mode=args.test_mode
    )

    benchmark.setup(
        chinese_bert_path=args.chinese_bert_path,
        english_bert_path=args.english_bert_path,
        load_base=not args.skip_base,
        load_rl=bool(args.rl_model_path)
    )

    test_data = load_test_data(args.test_file, args.num_samples)

    base_metrics, base_samples = None, None
    rl_metrics, rl_samples = None, None

    if not args.skip_base and benchmark.base_model:
        base_metrics, base_samples = benchmark.run_benchmark(
            benchmark.base_model,
            "Original Model",
            test_data,
            args.max_new_tokens,
            args.temperature
        )

    if benchmark.rl_model:
        rl_metrics, rl_samples = benchmark.run_benchmark(
            benchmark.rl_model,
            "RL Model",
            test_data,
            args.max_new_tokens,
            args.temperature
        )

    benchmark.print_results(base_metrics, rl_metrics)

    if args.show_samples > 0:
        benchmark.print_samples(base_samples, rl_samples, args.show_samples)

    if args.save:
        benchmark.save_results(
            args.output_dir,
            base_metrics, rl_metrics,
            base_samples, rl_samples
        )

    logger.info("Benchmark complete.")

if __name__ == "__main__":
    main()