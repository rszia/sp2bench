from minicons import scorer
import pandas as pd
import json
import numpy as np
import re
import tqdm
import os
import re
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_model_subdirs(models_dir, lg_prefix):
    # Find all subdirectories starting with lg_prefix
    all_subdirs = [
        os.path.join(models_dir, d)
        for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d)) and d.startswith(lg_prefix)
    ]
    return all_subdirs

def run_inference(benchmark_path, models, lan, exp, results_dir):
    benchmarks = [f for f in os.listdir(benchmark_path) if os.path.isfile(os.path.join(benchmark_path, f)) and 'json' in f and lan in f]

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    exp_folder = os.path.join(results_dir, exp)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)        
        
    results = []

    

    #exp = re.sub('(\.|/)+', '_', results_dir)

    for ckpt in tqdm.tqdm(models):
        
        try:        
            mlm_model = scorer.MaskedLMScorer(ckpt, 'cuda')
            
            mlm_eos = mlm_model.decode([mlm_model.eos_token_id])[0]
            

            model_name = os.path.basename(ckpt) ## Replace the re.sub("(\./|/)"... stuff
            model_name = re.sub('(_babyLM_TW_FR_models_|_models_|_spbpe_concat|FacebookAI_|_sp_concat)', '', model_name)

            model_ea = os.path.join(exp_folder, f"{model_name}_error_analysis/")
            if not os.path.exists(model_ea):
                os.makedirs(model_ea)

            for b in tqdm.tqdm(benchmarks):
                with open(os.path.join(benchmark_path, b), 'r', encoding = 'utf-8') as f:
                    f = f.read().split('\n')
                    data = [json.loads(e) for e in f if e.strip()]
            
                
                term = data[0]['linguistics_term']
                UID  = data[0]['UID']

                for i in range(len(data)):
                    
                    good = data[i]['sentence_good']
                    bad =  data[i]['sentence_bad']
                    
                    if args.filter:
                        good = re.sub('(@|\\*) *', '', good)
                        bad = re.sub('(@|\\*) *', '', bad)
                    
                    pair = [good, bad]
                    pair_scores = mlm_model.conditional_score([mlm_eos, mlm_eos], pair, PLL_metric=metric)
                    data[i]['good_conditional'] = round(pair_scores[0], 4)
                    data[i]['bad_conditional'] = round(pair_scores[1], 4)  
                    data[i]['conditional_acc'] = float(pair_scores[0] > pair_scores[1])
                                    
                    #pair_scores = mlm_model.sequence_score(pair, PLL_metric=metric)
                    #data[i]['good_sequence'] = pair_scores[0]
                    #data[i]['bad_sequence'] = pair_scores[1]  
                    #data[i]['sequence_acc'] = float(pair_scores[0] > pair_scores[1])             

                conditional_agg = np.mean([data[i]['conditional_acc'] for i in range(len(data))])
                #sequences_agg = np.mean([data[i]['sequence_acc'] for i in range(len(data))])
                
                with open(model_ea + b, 'w', encoding='utf-8') as outfile:
                    for entry in data:
                        json.dump(entry, outfile, ensure_ascii=False)
                        outfile.write('\n')          


                    out_dict = {'model': model_name, 'file_name': b, 
                                'linguistics_term': term, 'UID': UID, 
                                'cond_score': conditional_agg, 
                                #'seq_score': sequences_agg,
                                }
                    results.append(out_dict)

                    df = pd.DataFrame(results)
                    df.set_index('model')
                    df.to_csv(os.path.join(results_dir, f"{exp}_results.csv"))
        except Exception as e:
            print(f"Error during run_inference: {e}")
                
parser = argparse.ArgumentParser(description='')

parser.add_argument('-lg', action="store", dest="lg", default = 'zh', type=str)
parser.add_argument('-filter', action="store", dest="filter", default = 1, type=int)
parser.add_argument('-models_dir', action="store", dest="models_dir", default='models/models_cleaned', type=str)
parser.add_argument('-benchmark_path', action="store", dest="benchmark_path", default='data/data_benchmark_minimal_pair', type=str)
parser.add_argument('-results_dir', action="store", dest="results_dir", default = "results/results_minimal_pair", type=str)

args = parser.parse_args()

if args.lg == 'zh':
    metric = 'original'
elif args.lg in ['en', 'fr']:
    metric = 'within_word_l2r'
else:
    ValueError('The -lg argument should be one of "zh", "en", or "fr".')

tasks = ['disfl_comma']#, 'disfl', 'disfl_eosbos']

# Add toplines if needed
toplines = ["FacebookAI/xlm-roberta-large", "FacebookAI/xlm-roberta-base"]
models_list = get_model_subdirs(args.models_dir, args.lg) + toplines
print("MODELS_LIST:", *models_list, sep='\n')

for task in tasks:
    run_inference(os.path.join(args.benchmark_path, task), models_list, args.lg, f"{task}_{args.lg}", args.results_dir)