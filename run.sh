#! /bin/bash

MODE=$1

if [ "$MODE" == "--pretrain" ]; then

    python src/lm_base_creation.py -language zh -corpus_name spoken -epoch 64 -group_texts True -exp_tag 5e4 -patience 10 -learning_rate 5e-4 -batch_size 64
    # python src/lm_base_creation.py -language en -corpus_name spoken -epoch 64 -group_texts True -exp_tag 5e4 -patience 10 -learning_rate 5e-4 -batch_size 64
    # python src/lm_base_creation.py -language fr -corpus_name spoken -epoch 64 -group_texts True -exp_tag 5e4 -patience 10 -learning_rate 5e-4 -batch_size 64

    # python src/lm_base_creation.py -language zh -corpus_name mixed -epoch 64 -group_texts True -exp_tag 5e4 -patience 10 -learning_rate 5e-4 -batch_size 64
    # python src/lm_base_creation.py -language en -corpus_name mixed -epoch 64 -group_texts True -exp_tag 5e4 -patience 10 -learning_rate 5e-4 -batch_size 64
    # python src/lm_base_creation.py -language fr -corpus_name mixed -epoch 64 -group_texts True -exp_tag 5e4 -patience 10 -learning_rate 5e-4 -batch_size 64

    # python src/lm_base_creation.py -language zh -corpus_name wiki -epoch 64 -group_texts True -exp_tag 5e4 -patience 10 -learning_rate 5e-4 -batch_size 64
    # python src/lm_base_creation.py -language en -corpus_name wiki -epoch 64 -group_texts True -exp_tag 5e4 -patience 10 -learning_rate 5e-4 -batch_size 64
    # python src/lm_base_creation.py -language fr -corpus_name wiki -epoch 64 -group_texts True -exp_tag 5e4 -patience 10 -learning_rate 5e-4 -batch_size 64

    # python src/lm_base_creation.py -language fr_debugging -corpus_name spoken -epoch 64 -group_texts True -exp_tag 5e4 -patience 10 \
        # -learning_rate 5e-4 -batch_size 64 -vocab_size 1122

elif [ "$MODE" == "--finetune" ]; then

    python src/finetune_and_eval.py -models_dir ./models/models_cleaned_ft -task red -corpus mcdc -benchmark_file './data/data_benchmark_token_classification/benchmark_reduc_zh.csv' \
    -ckpt './models/models_cleaned/zh_spoken_sp_5e4' -label_column red -lge zh

# python src/finetune_and_eval.py -models_dir ./models/models_cleaned_ft -task red -corpus buckeye -benchmark_file './data/data_benchmark_token_classification/benchmark_en_updated_small.csv' \
#  -ckpt './models/models_cleaned/en_spoken_sp_5e4' -label_column red -lge en

# python src/finetune_and_eval.py -models_dir ./models/models_cleaned_ft -task red -corpus cid -benchmark_file './data/data_benchmark_token_classification/benchmark_reduc_fr_simple.csv' \
#  -ckpt './models/models_cleaned/fr_spoken_sp_5e4' -label_column red -lge fr

elif [ "$MODE" == "--zeroshot" ]; then
    
    python src.run_minicons.py

else
    echo "Usage: bash run.sh [--pretrain | --finetune | --zeroshot] "
    exit 1
fi