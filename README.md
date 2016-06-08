## Automated essay scoring experiments for the [Automated Essay Scoring Kaggle Competition](https://www.kaggle.com/c/asap-aes)
### Created by Adam Varga, 2016

features include lemma count, average modifier count per noun phrase, average word length, number of spelling mistakes

### Usage:
1. Feature extraction: `python extract_fea.py`
2. Feature selection, training and cross-validation: `python train_and_test.py`