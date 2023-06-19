# IMG

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

IMG: Invariant features Masks for domain Generalization, a model for improving out-of-domain generalization by learning invariant features.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation
We use conda to manage python packages.
```bash
git clone https://github.com/yourusername/IMG.git
cd IMG
conda env create -n IMG --file environment.yml
conda activate IMG
```

## Usage
### Training Models
Run the following command to start the training process:
```bash
# train a model on TweetEval dataset
python run_text_disentangled_classification.py --train_file dataset/sentiment/TweetEval_sentiment/train/tc_train.csv --validation_file dataset/sentiment/TweetEval_sentiment/valid/tc_valid.csv --model_name_or_path facebook/bart-large  --num_train_epochs 100 --output_dir checkpoints/bart_TweetEval --model_class BartForTokenAttentionSparseCLSJoint
```

```bash
# evaluate model on IMDB dataset
python run_text_disentangled_classification.py --train_file dataset/sentiment/original_imdb/test.csv --validation_file dataset/sentiment/original_imdb/test.csv --model_name_or_path checkpoints/bart_TweetEval  --output_dir checkpoints/out/ --model_class BartForTokenAttentionSparseCLSJoint --only_evaluation
```
## License

This project is licensed under the [MIT License](LICENSE).

