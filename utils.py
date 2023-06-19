import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random

from torch import nn
from sklearn.manifold import TSNE
from dataclasses import dataclass
from transformers import AutoTokenizer, M2M100Model, M2M100Tokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from models.modeling_m2m_100 import M2M100ForDisentangledRepresentation, M2M100ForSequenceClassification



def set_every_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def add_special_tokens_(model, tokenizer, special_tokens):
    # special_tokens = {'additional_special_tokens': ['<token1>', '<token2>']}
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = tokenizer.vocab_size
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)  # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
    return num_added_tokens

def compute_mask_overlap(first_model_path="checkpoints/sequenceCLS_sparse_m2m100_yahoo", second_model_path="checkpoints/sequenceCLS_sparse_m2m100_imdb"):
    first_model = M2M100ForDisentangledRepresentation.from_pretrained(first_model_path)
    second_model = M2M100ForDisentangledRepresentation.from_pretrained(second_model_path)
    tokenizer = AutoTokenizer.from_pretrained(first_model_path)

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    with torch.no_grad():
        first_outputs = first_model(**inputs)
        first_mask = first_outputs.disentangle_mask
        second_outputs = second_model(**inputs)
        second_mask = second_outputs.disentangle_mask
    first_sparse = torch.sum(first_mask == 0.) / first_mask.numel()
    second_sparse = torch.sum(second_mask == 0.) / second_mask.numel()
    print(first_sparse, second_sparse)
    first_mask[first_mask != 0.] = 1.
    second_mask[second_mask != 0.] = 1.
    #overlap = torch.sum(first_mask * second_mask) / first_mask.numel()
    overlap1 = torch.sum(first_mask * second_mask) / torch.sum(first_mask)
    overlap2 = torch.sum(first_mask * second_mask) / torch.sum(second_mask)
    print(overlap1, overlap2)
    

def get_token_embeddings(model_path="checkpoints/sequenceCLS_sparse_m2m100_yahoo10"):
    positive_reviews = [
        "many are good .",
        "but still good .",
        "and surprisingly good !",
        "very good movie .",
        "also really good .",
        "was very good .",
        "good viewing fun .",
        "very good film.",
        "both are good !",
        "very good direction ."
    ]
    negative_reviews = [
        "oh good grief .",
        "however not good .",
        "some good points ?",
        "no good matches .",
        "not even good .",
        "but not good .",
        "good camera angles ?",
        "sounds good huh ?",
        "not good enough .",
        "a potentially good story",
    ]
    model = M2M100ForDisentangledRepresentation.from_pretrained(model_path)
    tokenizer = M2M100Tokenizer.from_pretrained(model_path)

    token_id = 43292 # good --> 43292
    positive_embeddings = []
    negative_embeddings = []
    with torch.no_grad():
        for text in positive_reviews:
            inputs = tokenizer(text, return_tensors="pt")
            #print(inputs["input_ids"], inputs["input_ids"].shape, (inputs["input_ids"]==43292).nonzero(as_tuple=True))
            batch_idx, token_idx = (inputs["input_ids"] == 43292).nonzero(as_tuple=True)
            outputs = model(**inputs)
            embeddings = outputs.hidden_states
            batch_idx = batch_idx.tolist()[0]
            token_idx = token_idx.tolist()[0]
            positive_embeddings.append(embeddings[batch_idx][token_idx].numpy())
        for text in negative_reviews:
            inputs = tokenizer(text, return_tensors="pt")
            # print(inputs["input_ids"], inputs["input_ids"].shape, (inputs["input_ids"]==43292).nonzero(as_tuple=True))
            batch_idx, token_idx = (inputs["input_ids"] == 43292).nonzero(as_tuple=True)
            outputs = model(**inputs)
            embeddings = outputs.hidden_states
            batch_idx = batch_idx.tolist()[0]
            token_idx = token_idx.tolist()[0]
            negative_embeddings.append(embeddings[batch_idx][token_idx].numpy())
    positive_embeddings = np.stack(positive_embeddings, axis=0)
    negative_embeddings = np.stack(negative_embeddings, axis=0)
    #print(positive_embeddings.shape, negative_embeddings.shape)
    positive_embeddings_2d = TSNE(n_components=2, perplexity=6).fit_transform(positive_embeddings)
    negative_embeddings_2d = TSNE(n_components=2, perplexity=6).fit_transform(negative_embeddings)
    #print(positive_embeddings_2d, negative_embeddings_2d)

    all_points = np.concatenate((positive_embeddings_2d, negative_embeddings_2d), axis=0)
    #print(all_points)
    #print(all_points[:, 0], all_points[:, 1])
    df = pd.DataFrame()
    df["comp-1"] = all_points[:, 0]
    df["comp-2"] = all_points[:, 1]
    df["label"] = len(positive_reviews)*[1] + len(negative_reviews)*[0]
    sns_plot = sns.scatterplot(x="comp-1", y="comp-2", hue=df.label.tolist(),data=df)
    figure = sns_plot.get_figure()
    figure.savefig('scatter_GOOD.png', dpi=400)


#compute_mask_overlap()