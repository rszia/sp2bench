#!/usr/bin/env python
# coding: utf-8
import math
import pandas as pd
import numpy as np
import collections
import datasets
from datasets import Dataset, Value, ClassLabel, Features
from evaluate import load

import torch
from torch import nn

import transformers
#from transformers import RobertaTokenizerFast
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import Trainer
from transformers import EarlyStoppingCallback
#import seaborn as sns
import shutil
import os
import re
import argparse
import sentencepiece as spm

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def normalize_tokens(row, tok_column):
    tmp_tok = row[tok_column].lower()
    tmp_tok = tmp_tok.replace("'",'').replace('=','').replace('_','').replace('-','').replace('@@','*').replace('@','*').replace('#',",").replace('dummy',",")
    return tmp_tok

def addfold(spk,folds):
    for fold in folds.keys():
        if spk in folds[fold]:
            return fold

def small_data(df, keep, folds):
    folds[-1] = []
    targets = [x for x in folds.keys() if x > 0]
    for t in targets:
        folds[-1] = folds[-1] + folds[t][keep:] 
        folds[t] = folds[t][:keep]
        
    df['fold'] = df.apply(addfold,axis=1)
    df = df[df['fold'] > 0] 

    return df

def token2sent(df, tok_column, tokenizer, threshold=0.5):
    res = []
    
    tmp_toks = []
    tmp_labels = []

    for index,row in df.iterrows():
        if (row[tok_column] in ['#','dummy',',']) and row['duration'] > threshold:
            if tmp_toks != []:
                res.append([tmp_toks,tmp_labels,row['fold']])#
                tmp_toks = []
                tmp_labels = []
        else:
            tmp_toks.append(row[tok_column])
            tmp_labels.append(int(row['label']))
    output = pd.DataFrame(res,columns=[tok_column,'labels','fold'])
    output[tok_column] = output[tok_column].apply(lambda lst: [tokenizer.bos_token] + lst + [tokenizer.eos_token])
    output['labels'] = output["labels"].apply(lambda lst: [0] + lst + [0])
    return output

def concat_sequences(df, tok_column, max_length): ## Originally the compact() function
    res = []
    temp_toks = []
    temp_labels = []
    
    curr_fold = df.fold[0]
   
    for index,row in df.iterrows():
        if len(temp_toks) + len(row[tok_column]) <= max_length and row['fold'] == curr_fold:
            temp_toks = temp_toks + row[tok_column]
            temp_labels = temp_labels + row['labels']
        else:
            res.append([temp_toks,temp_labels,row['fold']])
            temp_toks = row[tok_column]
            temp_labels = row['labels']
        curr_fold = row['fold']
    print("*\n"*5, pd.DataFrame(res,columns=[tok_column,'labels','fold']))
    return pd.DataFrame(res,columns=[tok_column,'labels','fold'])

## Create HF datasets from the dataframes
def bin2bio(row,label_str):
    binlist = row['labels']
    res = []
    prev = 0
    for item in binlist:
        if item == 1 :
            if prev != 1:
                res.append('B-'+label_str) 
            else:
                res.append('I-'+label_str)
            prev = 1
        else:
            res.append('O')
            prev = 0
    return res

def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

def flatten_dict(d, parent_key='', sep='_'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items  

class RoBERTaFineTuner:
    def __init__(self, args, tokenizer, metric, folds, nb_folds):
        self.args = args
        self.tokenizer = tokenizer
        self.metric = metric
        self.folds = folds
        self.nb_folds = nb_folds
        
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.lr = args.lr
        self.ft_eps = args.ft_eps
        self.patience = args.patience
        self.freeze_to = args.freeze_to
        
        self.checkpoint = args.ckpt
        self.results_folder = args.results_dir
        self.models_folder = args.models_dir
        
        self.figs_folder = os.path.join(args.results_dir, os.path.basename(args.ckpt), "figs") # args.figs_dir
        self.logs_folder = os.path.join(args.results_dir, os.path.basename(args.ckpt), "logs") # re.sub("/$", "_logs/", args.results_dir)
        
        self.exp_tag = '_' + args.exp_tag if args.exp_tag != '' else ''


        for folder in [self.results_folder, self.figs_folder, self.models_folder, self.logs_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
    def tokenize_and_align_labels(self, examples, tok_column):
        tokenized_inputs = self.tokenizer(examples[tok_column], padding='max_length', truncation=True, is_split_into_words=True, add_special_tokens = False, max_length = self.max_length)
        
        tokenized_inputs['input_tokens'] = []
        for i in range(len(tokenized_inputs.input_ids)):
            tokens = [x for x in self.tokenizer.convert_ids_to_tokens(tokenized_inputs.input_ids[i]) if x not in self.tokenizer.all_special_tokens]
            tokenized_inputs['input_tokens'].append(tokens)  

        labels = []
        
        for i, label in enumerate(examples["tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            token_ids = tokenized_inputs.input_ids[i]
            previous_word_idx = None
            label_ids = []
            for word_idx in range(len(word_ids)):
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_ids[word_idx] is None or token_ids[word_idx] in self.tokenizer.all_special_ids:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_ids[word_idx] != previous_word_idx:
                    label_ids.append(label[word_ids[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    # Modified to make sure it follows the BIO scheme
                    if label[word_ids[word_idx]]==0:
                        good_label = 1
                    else:
                        good_label = label[word_ids[word_idx]]
                    label_ids.append(good_label)
                previous_word_idx = word_ids[word_idx]

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
        
    def prepare_compute_metric_with_labellist(self, label_list):
        def compute_metric(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)
                ]
        
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                    for prediction, label in zip(predictions, labels)
                ]

            results = self.metric.compute(predictions=true_predictions, references=true_labels)
            return {
                "f1": results["overall_f1"],
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "accuracy": results["overall_accuracy"],
            }
        return compute_metric
     
    def run_one_fold(self, task, fold, split_dataset, label_list, tok_column, weighted=False, verbose=True, error_analysis=False, keep_models=False):
        
        print(fold)
        tokenized_split_dataset = split_dataset.map(lambda d : self.tokenize_and_align_labels(d, tok_column), batched=True)
        print("start run_complete_expe")
        model = AutoModelForTokenClassification.from_pretrained(self.checkpoint, num_labels=len(label_list), trust_remote_code=True)
        # model_name = re.sub("(\./|/)", "_", self.checkpoint + self.exp_tag) +'-'+ str(fold)
        model_name = f"{os.path.basename(self.checkpoint)}{self.exp_tag}-{str(fold)}"
        
        print(model_name)
                 
        #layer freezing code goes here
        if self.freeze_to > 0.0:        

            for name, param in model.named_parameters():
                if 'embedding' in name:
                    param.requires_grad = False
                    print(name)
                elif 'embed_tokens' in name:
                    param.requires_grad = False
                    print(name)                
        
            freeze_to_layer = int(round(model.config.num_hidden_layers*self.freeze_to))
            freeze_to_layer   
            
            freeze_to_alpha = freeze_to_layer * 2
            
            for layer in range(0,freeze_to_layer):
                for name, param in model.named_parameters():
                    if 'encoder.layer.' + str(layer) + '.' in name:
                        param.requires_grad = False
                        print(name)
                    elif 'layers.' + str(layer) + '.' in name:
                        param.requires_grad = False
                        print(name)

            for alpha in range(0,freeze_to_alpha):
                for name, param in model.named_parameters():
                    if 'alphas.' + str(alpha) in name:
                        param.requires_grad = False
                        print(name)
             
        if verbose:
            print(model)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("Device in use:",device)
        
        #model.to('cuda')                                      #####
        args = TrainingArguments(
            os.path.join(self.models_folder, model_name+"-finetuned-"+task),
            eval_strategy = "epoch",
            #evaluation_strategy = "no",
            save_strategy ="epoch",
            logging_strategy="epoch",        
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.ft_eps,
            weight_decay=0.01,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model = 'f1',
            greater_is_better =True
            )

        
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        compute_metric = self.prepare_compute_metric_with_labellist(label_list)
        
        if weighted:
            trainer = CustomTrainer(model,args,
                              tokenized_split_dataset["train"],tokenized_split_dataset["valid"],
                              data_collator,self.tokenizer,compute_metrics=compute_metric,
                              callbacks = [EarlyStoppingCallback(early_stopping_patience=self.patience)]
                              )

        else:
            trainer = Trainer(model,args,
                              train_dataset=tokenized_split_dataset["train"],
                              eval_dataset=tokenized_split_dataset["valid"],
                              data_collator=data_collator,tokenizer=self.tokenizer,compute_metrics=compute_metric,
                              callbacks = [EarlyStoppingCallback(early_stopping_patience=self.patience)]
                              ) 
            
        print(args)
        trainer.train()
        trainer.save_state()
        
        predictions, labels, _ = trainer.predict(tokenized_split_dataset["test"])
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
            ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
            ]
        
        if error_analysis:
            EXP_FOLDER = os.path.join(self.logs_folder, f'error_analysis_{task}_{os.path.basename(self.checkpoint)}{self.exp_tag}')
            # EXP_FOLDER = self.logs_folder + '/error_analysis_' + task+"_"+ re.sub("(\./|/)", "_", self.checkpoint + self.exp_tag) + '/'
            if not os.path.exists(EXP_FOLDER):
                os.makedirs(EXP_FOLDER)
            EA_df = pd.DataFrame(tokenized_split_dataset["test"])
            EA_df['predict'] = true_predictions
            EA_df['gold'] = true_labels
            #EA_df.to_csv(EXP_FOLDER+"error_analysis_"+task+"_"+model_name+'.csv')
            EA_df.to_csv(EXP_FOLDER+"complex_EA_"+str(fold)+'.csv')    
            
            tokens_col = [item for row in EA_df['input_tokens'] for item in row]
            predict_col = [item for row in EA_df['predict'] for item in row]
            gold_col = [item for row in EA_df['gold'] for item in row]
            EA_df = pd.DataFrame(list(zip(tokens_col, predict_col, gold_col)),
                           columns =['token', 'prediction', 'gold'])        
            # EA_df.to_csv(EXP_FOLDER+'simple_EA_'+str(fold)+'.csv')
            EA_df.to_csv(os.path.join(EXP_FOLDER, 'simple_EA_'+str(fold)+'.csv'))
            

        if not keep_models:
            shutil.move(os.path.join(self.models_folder, model_name+"-finetuned-"+task, 'trainer_state.json'), EXP_FOLDER+'trainer_state_'+str(fold)+'.json')
            shutil.rmtree(os.path.join(self.models_folder, model_name+"-finetuned-"+task))

        return self.metric.compute(predictions=true_predictions, references=true_labels)

    def run_crossvalid(self, task, base_dataset, label_list, tok_column, weighted=False, verbose=True, error_analysis=False, keep_models=False):
        
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast)
        
        results = dict() # {'fs':[],'prec':[],'rec':[]}
        for i in range(1,self.nb_folds+1):
            print(i)
            split_ds = datasets.DatasetDict({
                'train': base_dataset.filter(lambda example: example["fold"] not in [i,(i%self.nb_folds+1)]),
                'test': base_dataset.filter(lambda example: example["fold"] == i),
                'valid': base_dataset.filter(lambda example: example["fold"] == i%self.nb_folds+1)
            })
            print(label_list)
            res = self.run_one_fold(task, i, split_ds, label_list, tok_column, weighted, verbose, error_analysis, keep_models)
            # print(res)
            # print("res", res, "\nhas type", type(res))
            
            ### Added this one
            flattened_res = flatten_dict(res)
            for key, value in flattened_res.items():
                results.setdefault(key, []).append(value)
            # for key, value in res.items():
            #     if key not in results:
            #         results[key] = value
            #     results[key].append(value)
            #     else:
                    
        return results

    def run_complete_expe(self, expe_name, ds, label_list, tok_column, weighted=False):

        print('====')
        print(expe_name)
        print('====')
        m_name = os.path.basename(self.checkpoint) + self.exp_tag
        # m_name = re.sub("(\./|/)", "_", self.checkpoint + self.exp_tag)
        print('running ' + str(m_name))
        res_cv = self.run_crossvalid(expe_name, ds, label_list, tok_column, 
                                    weighted=weighted, verbose=False, error_analysis=True, keep_models=False)
        print("RES_CV", "*\n"*3, res_cv)
        res_cv_df = pd.DataFrame(res_cv)
        res_cv_df['model'] = m_name
        res_cv_df.to_csv(os.path.join(self.results_folder, expe_name+'_'+m_name+'_cv.csv'))

        return 0

parser = argparse.ArgumentParser(description='')

#parser.add_argument('-overall_name', action="store", dest="overall_name", default = '', type=str)
parser.add_argument('-task', action="store", dest="task", default = "red", type=str)
parser.add_argument('-lge', action="store", dest="lge", default = "fr", type=str)
parser.add_argument('-corpus', action="store", dest="corpus",default = 'buckeye',type=str)
parser.add_argument('-benchmark_file', action="store", dest="benchmark_file",default = '',type=str)
parser.add_argument('-benchmark_sep', action="store", dest="benchmark_sep",default = ',',type=str)
parser.add_argument('-label_column', action="store", dest="label_column",default = 'red',type=str)
parser.add_argument('-tok_column', action="store", dest="tok_column",default = 'tok',type=str)
parser.add_argument('-spk_column', action="store", dest="spk_column",default = 'speaker',type=str)

parser.add_argument('-exp_tag', action="store", dest="exp_tag", default = '', type=str)
parser.add_argument('-ckpt', action="store", dest="ckpt", default = "", type=str)

parser.add_argument('-run_only', action="store", dest="run_only", default = 0, type=int)
parser.add_argument('-compact', action="store", dest="compact", default = True, type=boolean_string)
parser.add_argument('-voc_prune', action="store", dest="voc_prune", default = True, type=boolean_string)

parser.add_argument('-ft_eps', action="store", dest="ft_eps", default = 10, type=int)
parser.add_argument('-patience', action="store", dest="patience", default = 3, type=int)

parser.add_argument('-prom_cutoff', action="store", dest="prom_cutoff", default = 1.25, type=float)
parser.add_argument('-red_cutoff', action="store", dest="red_cutoff", default = 0.75, type=float)


parser.add_argument('-batch_size', action="store", dest="batch_size", default = 32, type=int)
parser.add_argument('-max_length', action="store", dest="max_length", default = 128, type=int)
parser.add_argument('-lr', action="store", dest="lr", default = 2e-5, type=float)

parser.add_argument('-freeze_to', action="store", dest="freeze_to", default = 0.0, type=float)
parser.add_argument('-results_dir', action="store", dest="results_dir", default = "./results/results_token_classification", type=str)
parser.add_argument('-models_dir', action="store", dest="models_dir", default = "./models/models_cleaned_ft", type=str)
# parser.add_argument('-figs_dir', action="store", dest="figs_dir", default = "./figs/", type=str)


args = parser.parse_args()

print("load tokenizer")
tokenizer = AutoTokenizer.from_pretrained(args.ckpt,max_len=args.max_length,add_prefix_space=True)

if args.corpus == 'cid':
	FOLDS = {1:['AB','CM'],2:['YM','AG'],3:['EB','SR'],4:['LL','NH'],
		     5:['BX','MG'],6:['AP','LJ'],7:['IM','ML'],8:['MB','AC']}
elif args.corpus == 'buckeye':
	#FOLDS = {1:['s01', 's02', 's06', 's03'], 2:['s04', 's05', 's11', 's10'], 3:['s08', 's07', 's13', 's19'],
	#	 4:['s09', 's14', 's15', 's22'], 5:['s12', 's16', 's28', 's23'], 6:['s21', 's17', 's30', 's24'],
	#	 7:['s26', 's18', 's32', 's29'], 8:['s31', 's20', 's33', 's35'], 9:['s37', 's25', 's34', 's36'],
	#	 10:['s39', 's27', 's40', 's38']} 
	FOLDS = {1:['s01', 's02', 's06', 's03', 's37'], 2:['s04', 's05', 's11', 's10', 's25'], 
             3:['s08', 's07', 's13', 's19', 's34'], 4:['s09', 's14', 's15', 's22', 's36'],
             5:['s12', 's16', 's28', 's23', 's39'], 6:['s21', 's17', 's30', 's24', 's27'], 
             7:['s26', 's18', 's32', 's29', 's40'], 8:['s31', 's20', 's33', 's35', 's38']}         
elif args.corpus == 'mcdc':
	FOLDS = {1:['MCDC_01'], 2:['MCDC_02'], 3:['MCDC_03'],
		 4:['MCDC_05'], 5:['MCDC_09'], 6:['MCDC_10'],
		 7:['MCDC_25'], 8:['MCDC_26']}		 
         
NB_FOLDS = len(FOLDS.keys())    

print("load metric")
metric = load("seqeval")

print("load read_csv")
df = pd.read_csv(args.benchmark_file, sep = args.benchmark_sep)

df[args.tok_column] = df[args.tok_column].fillna(',')
df[args.tok_column] = df.apply(lambda x: normalize_tokens(x, args.tok_column), axis=1) # Now normalize_tokens has 2 arguments
# df[tok_column] = df.apply(normalize_tokens,axis=1)
#df['duration'] = df['end']-df['start']

if args.label_column == "prom":
    df['label'] = df[args.label_column]> args.prom_cutoff
elif args.label_column == "red":
    df['label'] = df[args.label_column]< args.red_cutoff
elif args.label_column == "bc":
    df['label'] = df[args.label_column]

print("load fold")
df['fold'] = df.apply(lambda row: addfold(row['speaker'], FOLDS), axis=1)

if args.run_only > 0:
    df = small_data(df = df, keep = args.run_only, folds = FOLDS)
    
print("token2sent")
df_ready = token2sent(df, args.tok_column, tokenizer)

print("compact")
if args.compact:
    df_ready = concat_sequences(df_ready, args.tok_column, args.max_length)
    
df_ready['labels_bio'] = df_ready.apply(lambda row: bin2bio(row,args.task.upper()),axis=1)

print("dataset processing")
dataset = Dataset.from_pandas(df_ready)
dataset = dataset.map(lambda ex: {"tags": ex["labels_bio"]}) #red_bio
all_labels = get_label_list(dataset["tags"])
dataset = dataset.cast_column("tags", datasets.Sequence(datasets.ClassLabel(names=all_labels)))
label_list = dataset.features["tags"].feature.names

# Create the fine-tuner instance and run experiment
print("Creating fine-tuner and starting experiment")
fine_tuner = RoBERTaFineTuner(args, tokenizer, metric, FOLDS, NB_FOLDS)
fine_tuner.run_complete_expe(args.lge+'_'+args.task, dataset, label_list, args.tok_column)