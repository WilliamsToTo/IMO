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

def estimate_numOfTokens(file_path):
    data = pd.read_csv(file_path)
    data_str = data.to_string()
    tokens = word_tokenize(data_str)

    # Step 4: Count the tokens
    token_count = len(tokens)
    print("Estimated cost for ChatGPT:", 0.002 * token_count/1000)

def evaluate_chatgpt(file_path):
    def find_first_label(s):
        s = str(s.lower())
        if s is None:
            return 2
        labels = ['positive', 'negative', 'neutral']
        label_positions = []

        for label in labels:
            position = s.find(label)
            if position != -1:
                label_positions.append((label, position))

        if not label_positions:
            return 2

        label_positions.sort(key=lambda x: x[1])
        mapping = {'positive':1, 'negative':0, 'neutral':2}
        return mapping[label_positions[0][0]]

    df = pd.read_csv(file_path)
    df["alpaca_lora_predicts"] = df["alpaca_lora_response"].apply(find_first_label)
    print(accuracy_score(df["label"], df["alpaca_lora_predicts"]))
    df.to_csv(file_path, index=False)

def create_domain_classification_dataset(mode="train"):
    if mode == "train":
        domain_paths = {"amazon_polarity/tc_train.csv": "amazon",
                        "yelp_polarity/tc_train.csv": "yelp",
                        "TweetEval_sentiment/train/tc_train.csv": "TweetEval",
                        "original_imdb/train.csv": "imdb",
                        "filter_yahoo/train/tc_train.csv": "yahoo"}
    else:
        domain_paths = {"amazon_polarity/tc_test.csv": "amazon",
                        "yelp_polarity/tc_test.csv": "yelp",
                        "TweetEval_sentiment/test/tc_test.csv": "TweetEval",
                        "original_imdb/test.csv": "imdb",
                        "filter_yahoo/test/tc_test.csv": "yahoo"}

    combinations = list(itertools.combinations(domain_paths.items(), 2))

    # Print combinations
    for domain_comb in tqdm(combinations):
        domain1_path = domain_comb[0][0]
        domain1_name = domain_comb[0][1]
        domain2_path = domain_comb[1][0]
        domain2_name = domain_comb[1][1]

        domain1_dataframe = pd.read_csv(domain1_path)
        domain2_dataframe = pd.read_csv(domain2_path)

        # Add a new column 'domain' to each DataFrame
        domain1_dataframe['domain'] = domain1_name
        domain2_dataframe['domain'] = domain2_name

        # Select only 'text' and 'domain' columns
        new_domain1_dataframe = domain1_dataframe[['text', 'domain']]
        new_domain2_dataframe = domain2_dataframe[['text', 'domain']]

        # Concatenate the two DataFrames
        result = pd.concat([new_domain1_dataframe, new_domain2_dataframe])

        if mode == "train":
            # Count the number of occurrences of each label
            counts = result['domain'].value_counts()

            # Find the label with the smallest number of occurrences
            max_count = counts.max()

            # Create an empty DataFrame to store the balanced data
            balanced_result = pd.DataFrame(columns=['text', 'domain'])

            # Iterate over each label
            for label in counts.index:
                # Filter the original DataFrame to include only rows with this label
                label_result = result[result['domain'] == label]

                # If the number of rows with this label is less than the minimum count,
                # repeat the rows until the count is equal to the minimum count
                if len(label_result) < max_count:
                    num_repeats = int(max_count / len(label_result))
                    remainder = max_count % len(label_result)
                    repeated_result = pd.concat([label_result] * num_repeats + [label_result.sample(remainder)])
                else:
                    repeated_result = label_result

                # Append the repeated rows to the balanced DataFrame
                balanced_result = pd.concat([balanced_result, repeated_result])

            # Shuffle the rows of the balanced DataFrame
            balanced_result = balanced_result.sample(frac=1)

            # Write the result to a new CSV file
            balanced_result.rename(columns={"domain": "label"}, inplace=True)
            balanced_result.to_csv(f"domain_classification/{domain1_name}_{domain2_name}_train.csv", index=False)
        else:
            result.rename(columns={"domain": "label"}, inplace=True)
            result.to_csv(f"domain_classification/{domain1_name}_{domain2_name}_test.csv", index=False)

create_domain_classification_dataset(mode="test")

# evaluate_chatgpt(file_path="filter_yahoo/test/tc_test_alpacaResponse.csv")
# evaluate_chatgpt(file_path="TweetEval_sentiment/test/tc_test_alpacaResponse.csv")
# evaluate_chatgpt(file_path="original_imdb/tc_valid_alpacaResponse.csv")
# evaluate_chatgpt(file_path="yelp_polarity/tc_valid_alpacaResponse.csv")
# evaluate_chatgpt(file_path="amazon_polarity/tc_valid_alpacaResponse.csv")

# sample_data(input_file="amazon_polarity/tc_train.csv", output_file="amazon_polarity/tc_train_1M.csv", num_samples=1000000)

# split_dataset(dataset_file="amazon_polarity/tc_train.csv", training_file="amazon_polarity/tc_train.csv",
#               valid_file="amazon_polarity/tc_valid.csv", size_valid=1000)

# create_tc_dataset(file_path="amazon/train/train.txt", save_path="amazon/train/tc_train.csv")
# create_tc_dataset(file_path="amazon/valid/valid.txt", save_path="amazon/valid/tc_valid.csv")
# create_tc_dataset(file_path="amazon/test/test.txt", save_path="amazon/test/tc_test.csv")

# target_size=10
# create_transfer_target_domain_dataset(file_path="yelp_polarity/tc_train.csv", save_path=f"yelp_polarity/{target_size}_shot/tc_train_1.csv",
#                                       target_size=target_size)


# estimate_numOfTokens(file_path="filter_yahoo/test/tc_test.csv")
# estimate_numOfTokens(file_path="original_imdb/test.csv")
# estimate_numOfTokens(file_path="yelp_polarity/tc_test.csv")
# estimate_numOfTokens(file_path="amazon_polarity/tc_test.csv")
#estimate_numOfTokens(file_path="TweetEval_sentiment/test/tc_test.csv")

# process_TweetEval(file_path="TweetEval_sentiment/test/original_test.csv", save_file="TweetEval_sentiment/test/tc_test.csv")