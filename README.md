# IMG

## Table of Contents
- [Setup](#installation)
- [Usage](#usage)

## Setup
We use conda to manage python packages.
```bash
git clone https://github.com/yourusername/IMO.git
cd IMO
conda env create -n IMO --file environment.yml
conda activate IMO
```

## Usage
### Training and Evaluate Models
Run the following command to start the training process:
```bash
# train a model on TweetEval dataset
sbatch submit_job_example.sh
```

```bash
# evaluate model on IMDB dataset
python run_text_disentangled_classification.py --train_file dataset/sentiment/original_imdb/test.csv --validation_file dataset/sentiment/original_imdb/test.csv --model_name_or_path checkpoints/your_model  --output_dir checkpoints/out/ --model_class BartForTokenAttentionSparseCLSJoint_incremental --only_evaluation
```


