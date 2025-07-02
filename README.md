# sp2bench
WIP
This repo contains the source code for the **sp2bench** benchmark.

## Setup
TBA: requirements.txt
## How to run
### Pretrain a model from a given corpus 
```
bash run.sh --pretrain
```
Now you have a pretrained model. To evaluate the model's adaquecy on a given speech phenomenon, there are two strategies to be chosen from depending on the task:

- For tasks like *prominence* or *reduction*, the model will be given a sentence and asked whether each token exhibits the speech phenomenon. It is essentially a token classification task.
- For tasks like *discourse markers* or *fillers*, the benchmark is in the minimal pairs format. That is, one acceptable sentence (a genuine instance of the speech phenomenon) is paired with one unacceptable sentence (an unattested instance), and the model has to identify the acceptable one.

### Fine-tune the model
```
bash run.sh --finetune
```
### Zero-shot evaluation on the benchmark

```
bash run.sh --zeroshot
```

## Citation

[Extending the BabyLM Initiative : Promoting Diversity in Datasets and Metrics through High-Quality Linguistic Corpora](https://aclanthology.org/2024.conll-babylm.12/) (Prévot et al., CoNLL-BabyLM 2024)

[Spontaneous Speech Variables for Evaluating LLMs Cognitive Plausibility
](https://arxiv.org/abs/2505.16277) (Wang et al., CMCL 2025)

[Zero-Shot Evaluation of Conversational Language Competence in Data-Efficient LLMs Across English, Mandarin, and French](https://2025.sigdial.org/list-of-accepted-papers/) (Wang et al., SIGDIAL 2025)
