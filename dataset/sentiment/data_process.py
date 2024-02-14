import json
import random
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
import itertools
from tqdm import tqdm


def create_tc_dataset(file_path, save_path):
    file1 = open(file_path, 'r')
    lines = file1.readlines()
    pairs = []
    for line in lines:
        example = json.loads(line)
        review = example["review"]
        label = example["score"]
        pairs.append((review, label))
    tc_dataset = pd.DataFrame(pairs, columns=["sentence", "label"])
    tc_dataset.to_csv(save_path, index=False)


def create_transfer_target_domain_dataset(file_path, save_path, target_size):
    data = pd.read_csv(file_path)
    positive_pairs = []
    negative_pairs = []
    for index, row in data.iterrows():
        review = row["text"] # text or content
        label = row["label"]
        if label == 1:
            positive_pairs.append((review, label))
        else:
            negative_pairs.append((review, label))

    sampled_positive_pairs = random.sample(positive_pairs, int(target_size/2))
    sampled_negative_pairs = random.sample(negative_pairs, int(target_size/2))
    pairs = sampled_positive_pairs + sampled_negative_pairs
    tc_dataset = pd.DataFrame(pairs, columns=["text", "label"])
    tc_dataset.to_csv(save_path, index=False)

def split_dataset(dataset_file, training_file, valid_file, size_valid=1000):
    df = pd.read_csv(dataset_file)
    # get the number of rows in the DataFrame
    n_rows = df.shape[0]

    # randomly select 1000 rows for validation
    val_indices = random.sample(range(n_rows), size_valid)
    val_set = df.iloc[val_indices]

    # create the training set by removing the validation set
    train_set = df.drop(val_indices)

    # save the validation and training sets as CSV files
    val_set.to_csv(valid_file, index=False)
    train_set.to_csv(training_file, index=False)

def sample_data(input_file, output_file, num_samples):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Randomly sample num_samples rows
    df_sampled = df.sample(n=num_samples)

    # Write the sampled data to a new CSV file
    df_sampled.to_csv(output_file, index=False)


def process_TweetEval(file_path, save_file):
    df = pd.read_csv(file_path)
    # Remove rows with label 1 (neutral)
    df = df[df['label'] != 1]
    # Convert rows with label 2 to 1
    df['label'] = df['label'].replace(2, 1)
    # Save the modified DataFrame to a new CSV file
    df.to_csv(save_file, index=False)


# split_dataset(dataset_file="amazon_polarity/tc_train.csv", training_file="amazon_polarity/tc_train.csv",
#               valid_file="amazon_polarity/tc_valid.csv", size_valid=1000)

# create_tc_dataset(file_path="amazon/train/train.txt", save_path="amazon/train/tc_train.csv")
# create_tc_dataset(file_path="amazon/valid/valid.txt", save_path="amazon/valid/tc_valid.csv")
# create_tc_dataset(file_path="amazon/test/test.txt", save_path="amazon/test/tc_test.csv")


# process_TweetEval(file_path="TweetEval_sentiment/test/original_test.csv", save_file="TweetEval_sentiment/test/tc_test.csv")