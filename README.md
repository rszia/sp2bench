# sp2bench

Spontaneous human speech exhibits a rich diversity of linguistic phenomena, such as the use of fillers (e.g., "uh") and discourse markers. These phenomena, absent from written text, may serve as a rich source to uncover the language processing mechanisms of humans.

We present a benchmark covering a range of linguistic phenomena, extracted from high-quality spoken corpora in English, French, and Mandarin.

To turn linguistic phenomena into prediction tasks, we provide different procedures accomodate for the nature of each task:

- For tasks like *discourse markers* or *fillers*, the goal is to identify whether the placement of the discourse marker/filler is correct. Therefore, the benchmark is in the minimal pairs format -- one acceptable sentence (a genuine instance of the speech phenomenon) is paired with one unacceptable sentence (an unattested instance), and the model has to identify the acceptable one. You can follow the [zero-shot evaluation](#zeroshot) section below to run them.
- Certain tasks, such as *prominence* or *reduction*, are better suited as token classification tasks. Here, the model will be given a sentence and asked whether each token exhibits the speech phenomenon. To evaluate on these tasks, it's needed to fine-tune the model beforehand. You can find an example in the [fine-tuning evaluation](#finetuning).

## Setup
TBA: requirements.txt
## Pretraining a model from a given corpus 
```
bash run.sh --pretrain
```
See example usage in `run.sh`. The pretraining corpus is in the folder `./data/data_cleaned_txt/<language>`. You can specify the language to be en/zh/fr. 

## <a name="zeroshot"></a> Zero-shot evaluation

```
bash run.sh --zeroshot
```

## <a name="finetune"></a> Fine-tuning the model
```
bash run.sh --finetune
```

## Citation

For the context about BabyLM and fine-tuning experiments:
- [Extending the BabyLM Initiative : Promoting Diversity in Datasets and Metrics through High-Quality Linguistic Corpora](https://aclanthology.org/2024.conll-babylm.12/) (Prévot et al., CoNLL-BabyLM 2024)

- [Spontaneous Speech Variables for Evaluating LLMs Cognitive Plausibility
](https://arxiv.org/abs/2505.16277) (Wang et al., CMCL 2025)

For BLiMP-style, zero-shot experiments:

- [Zero-Shot Evaluation of Conversational Language Competence in Data-Efficient LLMs Across English, Mandarin, and French](https://2025.sigdial.org/list-of-accepted-papers/) (Wang et al., SIGDIAL 2025)