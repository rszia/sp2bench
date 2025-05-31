import torch
from pathlib import Path
import tqdm as notebook_tqdm
import os
import re
import argparse
import json

import sentencepiece as spm

from tokenizers.processors import BertProcessing

from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from transformers import EarlyStoppingCallback, IntervalStrategy

from transformers import convert_slow_tokenizer
from transformers import PreTrainedTokenizerFast
from transformers import XLMRobertaTokenizerFast, XLMRobertaTokenizer
from transformers import XLMRobertaConfig, XLMRobertaForMaskedLM
from transformers import AutoTokenizer

from datasets import load_dataset

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

class RoBERTaTrainer:
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Set up paths and names
        self._setup_paths_and_names()
        
    def _setup_paths_and_names(self):
        """Setup experiment paths and model names"""
        if self.args.exp_tag != '':
            self.exp_tag = '_' + self.args.exp_tag
        else:
            self.exp_tag = ''
            
        self.model_name = f"{self.args.language}_{self.args.corpus_name}_{self.args.tok_name}{self.exp_tag}"
        self.save_path = os.path.join(self.args.model_path, self.model_name)
        
        self.train_path = os.path.join(self.args.text_path, self.args.language, f"{self.args.corpus_name}_train_sample.txt")
        self.valid_path = os.path.join(self.args.text_path, self.args.language, f"{self.args.corpus_name}_dev_sample.txt")
        
        # Create directories
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            os.makedirs(os.path.join(self.save_path, 'sp'))
    
    def tokenize_function_concat(self, sample):
        """Tokenization function for concatenated version (no padding, no truncation)"""
        return self.tokenizer(
            sample['text'],
            padding=False,
            truncation=False,
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        )

    def tokenize_function_base(self, sample):
        """Base tokenization function"""
        return self.tokenizer(
            sample['text'],
            padding=False,
            truncation=True,
            return_special_tokens_mask=True,
            return_token_type_ids=False,
        )

    def group_texts(self, examples, block_size=512):
        """Group texts into chunks of block_size"""
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop,
        # you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result
    
    def train_sentencepiece_tokenizer(self):
        """Train SentencePiece tokenizer"""
        print("Training SentencePiece tokenizer...")
        
        spm.SentencePieceTrainer.train(
            input=self.train_path,
            model_prefix=self.model_name,
            vocab_size=self.args.vocab_size,
            model_type=self.args.tok_type,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=["<mask>"]
        )

        # SentencePiece trainer does not allow to specify an output folder...
        os.rename(f"{self.model_name}.model", os.path.join(self.save_path, 'sp', f"{self.model_name}.model"))
        os.rename(f"{self.model_name}.vocab", os.path.join(self.save_path, 'sp', f"{self.model_name}.vocab"))

        sp = spm.SentencePieceProcessor(model_file=os.path.join(self.save_path, 'sp', f'{self.model_name}.model'))
        sp.vocab_file = os.path.join(self.save_path, 'sp', f'{self.model_name}.model')

        self.tokenizer = XLMRobertaTokenizer(
            vocab_file=sp.vocab_file, 
            max_len=512,
            clean_up_tokenization_spaces=False,
            return_special_tokens=True
        )
        
        self.tokenizer.save_pretrained(self.save_path)
        
    def create_model(self):
        """Create XLM-RoBERTa model"""
        print("Creating XLM-RoBERTa model...")
        
        config = XLMRobertaConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=514,
            max_seq_length=128,
            hidden_size=self.args.hidden_size,
            num_hidden_layers=self.args.num_hidden_layers,
            num_attention_heads=self.args.num_attention_heads,
            intermediate_size=self.args.intermediate_size,
            attention_probs_dropout_prob=self.args.attention_probs_dropout_prob,
            type_vocab_size=1,
        )

        self.model = XLMRobertaForMaskedLM(config)
        self.model.to(self.device)
        
        # Update tokenizer for dataset processing
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.save_path,
            max_len=512,
            clean_up_tokenization_spaces=False
        )
    
    def prepare_datasets(self):
        """Prepare training and validation datasets"""
        print("Preparing datasets...")
        
        # Load dataset
        if self.args.use_valid_data:
            dataset = load_dataset("text", data_files={
                "train": [self.train_path], 
                "valid": [self.valid_path]
            })
        else:
            dataset = load_dataset("text", data_files={"train": [self.train_path]})

        if self.args.group_texts:
            # Concatenated version
            tokenized_train = dataset['train'].map(
                self.tokenize_function_concat,
                batched=True,
                remove_columns=['text'],
                load_from_cache_file=False,
            )
            
            if self.args.use_valid_data:
                tokenized_valid = dataset['valid'].map(
                    self.tokenize_function_concat,
                    batched=True,
                    remove_columns=['text'],
                    load_from_cache_file=False,
                )

            tokenized_train = tokenized_train.map(
                self.group_texts,
                batched=True,
                batch_size=1000,
                num_proc=1,
                load_from_cache_file=False,
            )
            
            if self.args.use_valid_data:
                tokenized_valid = tokenized_valid.map(
                    self.group_texts,
                    batched=True,
                    batch_size=1000,
                    num_proc=1,
                    load_from_cache_file=False,
                )
        else:
            # Not concatenated version
            tokenized_train = dataset['train'].map(
                self.tokenize_function_base,
                batched=True,
                remove_columns=['text'],
                load_from_cache_file=False,
            )
            
            if self.args.use_valid_data:
                tokenized_valid = dataset['valid'].map(
                    self.tokenize_function_base,
                    batched=True,
                    remove_columns=['text'],
                    load_from_cache_file=False,
                )

        self.tokenized_train = tokenized_train
        if self.args.use_valid_data:
            self.tokenized_valid = tokenized_valid
    
    def setup_trainer(self):
        """Setup the Trainer"""
        print("Setting up trainer...")
        
        # Create Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )

        if self.args.use_valid_data:
            training_args = TrainingArguments(
                output_dir=self.save_path,
                overwrite_output_dir=True,
                num_train_epochs=self.args.epoch,
                per_device_train_batch_size=self.args.batch_size,
                per_device_eval_batch_size=self.args.batch_size,
                load_best_model_at_end=True,
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                save_total_limit=self.args.save_total_limit,
                prediction_loss_only=True,
                save_only_model=True,
                learning_rate=self.args.learning_rate,
                warmup_ratio=self.args.warmup_ratio,
                weight_decay=0.01,
                log_level="debug"
            )

            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=self.tokenized_train,
                eval_dataset=self.tokenized_valid,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=self.args.patience)]
            )
        else:
            training_args = TrainingArguments(
                output_dir=self.save_path,
                overwrite_output_dir=True,
                num_train_epochs=self.args.epoch,
                per_device_train_batch_size=self.args.batch_size,
                evaluation_strategy="no",
                save_strategy="epoch",
                logging_strategy="epoch",
                save_total_limit=self.args.save_total_limit,
                prediction_loss_only=True,
                save_only_model=True,
                learning_rate=self.args.learning_rate,
                warmup_ratio=self.args.warmup_ratio,
                weight_decay=0.01,
            )

            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=self.tokenized_train,
            )
    
    def train(self):
        """Run the complete training pipeline"""
        print(f"Starting training for model: {self.model_name}")
        print(f"Save path: {self.save_path}")
        print(f"Arguments: {self.args}")
        
        self.train_sentencepiece_tokenizer()
        self.create_model()
        self.prepare_datasets()
        self.setup_trainer()
        print("Starting model training...")
        self.trainer.save_model()
        self.trainer.save_state()
        
        self.trainer.train()

        self.trainer.save_model()
        self.trainer.save_state()
        self.trainer.save_model(self.save_path)
        
        print("Training completed!")

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-exp_tag', action="store", dest="exp_tag", default='', type=str)
    parser.add_argument('-language', action="store", dest="language", default='zh_debugging', type=str)
    parser.add_argument('-corpus_name', action="store", dest="corpus_name", default='spoken', type=str)
    parser.add_argument('-tok_name', action="store", dest="tok_name", default='sp', type=str)
    parser.add_argument('-tok_type', action="store", dest="tok_type", default='unigram', type=str)

    parser.add_argument('-save_total_limit', action="store", dest="save_total_limit", default=3, type=int)
    parser.add_argument('-use_valid_data', action="store", dest="use_valid_data", default=True, type=boolean_string)

    parser.add_argument('-epoch', action="store", dest="epoch", default=128, type=int)
    parser.add_argument('-batch_size', action="store", dest="batch_size", default=32, type=int)
    parser.add_argument('-vocab_size', action="store", dest="vocab_size", default=10000, type=int)

    parser.add_argument('-hidden_size', action="store", dest="hidden_size", default=512, type=int)
    parser.add_argument('-num_hidden_layers', action="store", dest="num_hidden_layers", default=4, type=int)
    parser.add_argument('-num_attention_heads', action="store", dest="num_attention_heads", default=8, type=int)
    parser.add_argument('-intermediate_size', action="store", dest="intermediate_size", default=2048, type=int)
    parser.add_argument('-hidden_dropout_prob', action="store", dest="hidden_dropout_prob", default=0.1, type=float)
    parser.add_argument('-attention_probs_dropout_prob', action="store", dest="attention_probs_dropout_prob", default=0.1, type=float)

    parser.add_argument('-text_path', action="store", dest="text_path", default="./data/data_cleaned_txt/", type=str)

    parser.add_argument('-warmup_ratio', action="store", dest="warmup_ratio", default=0.1, type=float)

    parser.add_argument('-model_path', action="store", dest="model_path", default="./models/models_cleaned/", type=str)
    parser.add_argument('-patience', action="store", dest="patience", default=10, type=int)
    parser.add_argument('-learning_rate', action="store", dest="learning_rate", default=2e-4, type=float)
    parser.add_argument('-group_texts', action="store", dest="group_texts", default=False, type=boolean_string)

    args = parser.parse_args()
    
    # Create and run trainer
    trainer = RoBERTaTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()