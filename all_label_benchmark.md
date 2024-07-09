# Benchmarks using all 3 subcategories if available

- B = 1
- T = 512
- lr = 1e-4, 3e-4, 6e-4, 1e-5

- train-score / test-score
- f1 per class

## COIN i3c
base 8 layer decoder config, no pretraining
.86 / .69 epoch 3
full log at hyp_cls/COIN-i3c-base_all-labels/

## DeBERTa base
.88 / .68 epoch 7 
full log at hyp_cls/DeBERTa-base_all-labels/

## RoBERTa base
.79 / .66 epoch 6
full log at hyp_cls/RoBERTa-base_all-labels/

## BERT base
.80 / .67 epoch 8
full log at hyp_cls/BERT-base_all-labels/

