import torch
import pdb
import os
import pandas as pd
from tqdm import tqdm
from torch_geometric.utils import degree
import numpy as np
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from torch_geometric.datasets import Reddit
from scipy import special
from bs4 import BeautifulSoup
import gc
from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data, Dataset
from settings import Settings
from accountant import get_priv
import string
import fasttext as ft


'''
Graph subsampling.
'''
def subsample_graph(data, rate=0.1, maintain_class_dists=True,
                    every_class_present=True):
    '''
    Given a data object, sample the graph based on the provided rate
    (as a percent)
    every_class_present: making sure that all classes are present in the
                         subsample (only if class distributions are not
                         maintained)
    '''
    if not 1 > rate > 0:
        raise Exception("Rate of subsampling graph must be in interval (0,1).")

    if maintain_class_dists:
        class_counts = torch.bincount(data.y[data.train_mask])
        new_class_counts = torch.floor_divide(class_counts, 1/rate).long()
        all_new_class_indexes = []
        for cls_val in range(class_counts.shape[0]):
            full_class_indexes = (data.y == cls_val).nonzero().squeeze()
            train_class_indexes = torch.tensor(np.intersect1d(full_class_indexes.numpy(), data.train_mask.nonzero().squeeze().numpy()))
            sample_idx_tensor = torch.randperm(
                    train_class_indexes.shape[0])[:new_class_counts[cls_val]]
            new_class_indexes = train_class_indexes[sample_idx_tensor]
            all_new_class_indexes.append(new_class_indexes)
        sample_tensor = torch.cat(all_new_class_indexes)
    else:
        if every_class_present:
            class_counts = torch.bincount(data.y[data.train_mask])
            new_class_counts = torch.floor_divide(class_counts, 1/rate).long()
            idx_from_every_class = []
            for cls_val in range(class_counts.shape[0]):
                full_class_indexes = (data.y == cls_val).nonzero().squeeze()
                train_class_indexes = torch.tensor(np.intersect1d(full_class_indexes.numpy(), data.train_mask.nonzero().squeeze().numpy()))
                sample_idx_tensor = torch.randperm(
                        train_class_indexes.shape[0]
                        )[:new_class_counts[cls_val]]
                new_class_indexes = train_class_indexes[sample_idx_tensor]
                idx_from_every_class.append(new_class_indexes[0].item())

            full_len = data.x[data.train_mask].shape[0]
            sample_len = int(full_len * rate)
            sample_tensor = torch.randperm(full_len)[:sample_len]

            # Adding indexes from each class to the sample tensor:
            sample_tensor = torch.cat(
                    (sample_tensor,
                     torch.tensor(idx_from_every_class))
                    ).unique()
        else:
            full_len = data.x[data.train_mask].shape[0]
            sample_len = int(full_len * rate)
            sample_tensor = torch.randperm(full_len)[:sample_len]

    val_idxs = data.val_mask.nonzero().squeeze()
    test_idxs = data.test_mask.nonzero().squeeze()
    sample_tensor = torch.cat((sample_tensor, val_idxs, test_idxs))

    data.x = data.x[sample_tensor]
    data.train_mask = data.train_mask[sample_tensor]
    data.val_mask = data.val_mask[sample_tensor]
    data.test_mask = data.test_mask[sample_tensor]
    data.y = data.y[sample_tensor]

    old_to_new_node_idx = {old_idx.item(): new_idx
                           for new_idx, old_idx in enumerate(sample_tensor)}

    # Updating adjacency matrix
    new_edge_index_indexes = []
    for idx in tqdm(range(data.edge_index.shape[1])):
        if (data.edge_index[0][idx] in sample_tensor) and \
           (data.edge_index[1][idx] in sample_tensor):
            new_edge_index_indexes.append(idx)

    new_edge_idx_temp = torch.index_select(
            data.edge_index, 1, torch.tensor(new_edge_index_indexes)
            )
    new_edge_idx_0 = [old_to_new_node_idx[new_edge_idx_temp[0][a].item()]
                      for a in range(new_edge_idx_temp.shape[1])]
    new_edge_idx_1 = [old_to_new_node_idx[new_edge_idx_temp[1][a].item()]
                      for a in range(new_edge_idx_temp.shape[1])]
    data.edge_index = torch.stack((torch.tensor(new_edge_idx_0),
                                   torch.tensor(new_edge_idx_1)))


def make_small_reddit(rate=0.1, maintain_class_dists=True):
    ss = Settings()
    root_dir = ss.root_dir
    data_collated = Reddit(os.path.join(root_dir, 'Reddit'))
    data = data_collated[0]
    subsample_graph(data, rate=rate,
                    maintain_class_dists=maintain_class_dists)
    out_dir = os.path.join(root_dir, "RedditS", "processed")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, "data.pt")
    torch.save(data, out_path)


'''
Mini-batching code
'''
def random_graph_split(data, n_subgraphs=10):
    '''
    Divide a graph into subgraphs using a random split:
        For n subsets, place nodes into subsets then for each node pair in
        the subgraph, check whether an edge exists in the original graph
    Note: Only the training portion of the graph is considered, val/test
          portions can be used as before with the original 'data' object
          with data.val_mask and data.test_mask
    '''
    full_len = data.x.shape[0]
    sample_tensor = torch.arange(full_len)[data.train_mask]
    sample_tensor = sample_tensor[torch.randperm(sample_tensor.size()[0])]

    batch_indexes = np.array_split(sample_tensor, n_subgraphs)

    batch_masks = []
    for idx_list in batch_indexes:
        batch_mask = torch.zeros(full_len, dtype=torch.bool)
        batch_mask[idx_list] = True
        batch_masks.append(batch_mask)

    return batch_masks


'''
Getting stats on nodes and edges.
'''

def get_avg_nodes_and_edges(filename):
    '''
    Given raw data with different subsample sizes and information on the number of
    nodes and edges for each, computes the average for each sample size and each
    dataset.
    '''
    df = pd.read_csv(filename)
    datasets = ['citeseer', 'cora', 'pubmed', 'reddit-small']
    subsample_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    avg_nodes = {}
    avg_edges = {}
    for dataset in datasets:
        for subsamp in subsample_sizes:
            subset = df[(df['dataset'] == dataset) & (df['subsample_size'] == subsamp)]
            avg_edges[(dataset, subsamp)] = subset['num_edges'].mean()
            if dataset != 'reddit-small':
                avg_nodes[(dataset, subsamp)] = subset['num_nodes'].mean() - 1500
            else:
                avg_nodes[(dataset, subsamp)] = subset['num_nodes'].mean() - 8026
                # subtracting the val+train mask amounts

    with open('temp_nodes.csv', 'w') as f:
        for key, value in avg_nodes.items():
            f.write(f"{key[0]},{key[1]},{value}\n")

    return avg_nodes, avg_edges


def get_train_edge_count(data, split_graph=False):
    '''
    Counts the number of edges used only in the training subset of the graph.
    '''
    if split_graph:
        train_nodes = data.batch_masks[0].nonzero().squeeze()
    else:
        train_nodes = data.train_mask.nonzero().squeeze()
    test_nodes = data.test_mask.nonzero().squeeze()
    edges = data.edge_index

    num_train_edges = 0
    num_test_edges = 0
    for idx in range(edges.shape[1]):
        edge = edges[:, idx]
        if edge[0] in train_nodes and edge[1] in train_nodes:
            num_train_edges += 1
        elif edge[0] in test_nodes and edge[1] in test_nodes:
            num_test_edges += 1

    return num_train_edges, num_test_edges


"""
Pokec pre-processing.
"""

def preprocess_pokec_dataset_df(rootdir, output_stats=False):
    '''
    Takes the raw user profiles as a dataframe and outputs just the 10,000 most
    completed users that like cats and dogs.
    Optionally outputs some statistics on the full data as well.
    '''
    filename = 'soc-pokec-profiles.txt'
    dirname = os.path.join(rootdir, 'Pokec', 'raw')
    fullname = os.path.join(dirname, filename)
    df = pd.read_csv(fullname, sep='\t', header=0)
    df = df.drop('Unnamed: 59', 1)
    df.columns = ['user_id', 'public', 'completion_percentage', 'gender',
                  'region', 'last_login', 'registration', 'AGE', 'body',
                  'I_am_working_in_field', 'spoken_languages', 'hobbies',
                  'I_most_enjoy_good_food', 'pets', 'body_type', 'my_eyesight',
                  'eye_color', 'hair_color', 'hair_type',
                  'completed_level_of_education', 'favourite_color',
                  'relation_to_smoking', 'relation_to_alcohol',
                  'sign_in_zodiac', 'on_pokec_i_am_looking_for',
                  'love_is_for_me', 'relation_to_casual_sex',
                  'my_partner_should_be', 'marital_status', 'children',
                  'relation_to_children', 'I_like_movies',
                  'I_like_watching_movie', 'I_like_music',
                  'I_mostly_like_listening_to_music',
                  'the_idea_of_good_evening',
                  'I_like_specialties_from_kitchen', 'fun',
                  'I_am_going_to_concerts', 'my_active_sports',
                  'my_passive_sports', 'profession', 'I_like_books',
                  'life_style', 'music', 'cars', 'politics', 'relationships',
                  'art_culture', 'hobbies_interests', 'science_technologies',
                  'computers_internet', 'education', 'sport', 'movies',
                  'travelling', 'health', 'companies_brands', 'more']

    notnans = [df[col][df[col].notna()].shape[0] for col in df]
    uniques = [df[col].dropna().unique().shape[0] for col in df]

    valid_cats = ['macka', 'mam macku', 'kocur']
    valid_dogs = ['pes', 'mam psa', 'mam psika', 'mam psov']

    cats = df[df['pets'].isin(valid_cats)]
    dogs = df[df['pets'].isin(valid_dogs)]

    sorted_num_null_cats = cats.apply(
            lambda row: row.isna().sum(), axis=1).sort_values()
    sorted_num_null_dogs = dogs.apply(
            lambda row: row.isna().sum(), axis=1).sort_values()

    sorted_cats = cats.reindex(sorted_num_null_cats.index)
        # 15597
    sorted_dogs = dogs.reindex(sorted_num_null_dogs.index)
        # 134465

    # Subsampled and nans removed:
    sorted_cats = sorted_cats.iloc[:10000].replace(np.nan, '', regex=True)
    sorted_dogs = sorted_dogs.iloc[:10000].replace(np.nan, '', regex=True)

    # html columns: fun, life_style and onwards
    html_col = ['fun', 'life_style', 'music', 'cars', 'politics',
                'relationships', 'art_culture', 'hobbies_interests',
                'science_technologies', 'computers_internet', 'education',
                'sport', 'movies', 'travelling', 'health', 'companies_brands',
                'more']
    for col in html_col:
        sorted_cats[col] = sorted_cats[col].apply(
                lambda x: BeautifulSoup(x, 'html.parser').text)
        sorted_dogs[col] = sorted_dogs[col].apply(
                lambda x: BeautifulSoup(x, 'html.parser').text)

    sorted_cats.to_csv(os.path.join(dirname, 'sorted_cats.csv'), index=False,
                       sep='\t')
    sorted_dogs.to_csv(os.path.join(dirname, 'sorted_dogs.csv'), index=False,
                       sep='\t')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cat_users = torch.tensor([int(user_id)
                              for user_id in sorted_cats['user_id']]).to(device)
    dog_users = torch.tensor([int(user_id)
                              for user_id in sorted_dogs['user_id']]).to(device)

    torch.save(cat_users, os.path.join(dirname, 'cat_users.pt'))
    torch.save(dog_users, os.path.join(dirname, 'dog_users.pt'))

    if output_stats:
        with open('pokec_stats.csv', 'w') as nn_f:
            nn_f.write('Column_name,')
            for col in df.columns:
                nn_f.write(col)
                nn_f.write(',')
            nn_f.write('\n')
            nn_f.write('Num_not_null,')
            for notnan in notnans:
                nn_f.write(str(notnan))
                nn_f.write(',')
            nn_f.write('\n')
            nn_f.write('Num_unique,')
            for unique in uniques:
                nn_f.write(str(unique))
                nn_f.write(',')

        plt.figure()
        (df['completion_percentage'] / 100).hist(alpha=0.5, bins=50, weights=np.ones(df.shape[0]) / df.shape[0])
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel("Completion Percentage")
        plt.show()


def prepare_pokec_bows_embeddings():
    data_dir = 'data'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(data_dir, 'sorted_cats.csv'), 'r') as f:
        read = [line.strip('\n').split('\t') for line in f.readlines()]
        colnames = read[0]
        cat_lines = read[1:]

    with open(os.path.join(data_dir, 'sorted_dogs.csv'), 'r') as f:
        read = [line.strip('\n').split('\t') for line in f.readlines()]
        dog_lines = read[1:]

    cat_lines = [line[9:-1] for line in cat_lines]
    dog_lines = [line[9:-1] for line in dog_lines]

    split_cat_lines = [[word.lower().translate(
                        str.maketrans('', '', string.punctuation)
                        ) for line in user for word in line.split()]
                        for user in cat_lines]
    split_dog_lines = [[word.lower().translate(
                        str.maketrans('', '', string.punctuation)
                        ) for line in user for word in line.split()]
                        for user in dog_lines]

    both_lines = split_cat_lines + split_dog_lines

    freq = {}
    idx_to_word = {}
    word_to_idx = {}
    num_words = 0
    for user in both_lines:
        for word in user:
            if word in freq.keys():
                freq[word] += 1
            else:
                freq[word] = 1
                word_to_idx[word] = num_words
                idx_to_word[num_words] = word
                num_words += 1

    cat_indexes = [[word_to_idx[word] for word in user] for user in split_cat_lines]
    dog_indexes = [[word_to_idx[word] for word in user] for user in split_dog_lines]

    new_cat_indexes = []
    max_threshold = 15000  # 15000
    min_threshold = 15  # 15
    kept_word_idxs = []
    for user in cat_indexes:
        new_user = []
        for idx in user:
            if min_threshold < freq[idx_to_word[idx]] <= max_threshold:
                new_user.append(idx)
                kept_word_idxs.append(idx)
        new_cat_indexes.append(new_user)

    new_dog_indexes = []
    for user in dog_indexes:
        new_user = []
        for idx in user:
            if min_threshold < freq[idx_to_word[idx]] <= max_threshold:
                new_user.append(idx)
                kept_word_idxs.append(idx)
        new_dog_indexes.append(new_user)

    adjusted_num_words = len(list(set(kept_word_idxs)))
    old_to_new_idx = {idx: i for (idx, i) in zip(list(set(kept_word_idxs)),
                                                 range(adjusted_num_words))}

    cat_indexes = [[old_to_new_idx[idx] for idx in user] for user in new_cat_indexes]
    dog_indexes = [[old_to_new_idx[idx] for idx in user] for user in new_dog_indexes]
    new_word_to_idx = {word: old_to_new_idx[idx] for word, idx in word_to_idx.items() if idx in old_to_new_idx.keys()}
    new_idx_to_word = {old_to_new_idx[idx]: word for idx, word in idx_to_word.items() if idx in old_to_new_idx.keys()}

    all_cat_embeds = torch.zeros(len(cat_lines), adjusted_num_words).to(device)
    all_dog_embeds = torch.zeros(len(dog_lines), adjusted_num_words).to(device)

    for ex_idx, (cat_user, dog_user) in enumerate(zip(cat_indexes, dog_indexes)):
        for word_idx in cat_user:
            all_cat_embeds[ex_idx, word_idx] = 1
        for word_idx in dog_user:
            all_dog_embeds[ex_idx, word_idx] = 1

    print("Size of embeddings:", all_cat_embeds.element_size() * all_cat_embeds.nelement())
    print("Original number of words:", num_words)
    print("Adjusted number of words:", adjusted_num_words)

    torch.save(all_cat_embeds, os.path.join(data_dir, 'cat_embeds_bows.pt'))
    torch.save(all_dog_embeds, os.path.join(data_dir, 'dog_embeds_bows.pt'))


def prepare_pokec_fasttext_embeddings():
    '''
    Final dims of cat/dog embeddings: 10000x300 (all word embeds for a user are averaged)
    '''
    embed_dim = 300
    data_dir = 'data'
    ft_dir = 'cc.sk.300.bin'

    device = torch.device('cuda' if torch.cuda.is_available()
                               else 'cpu')

    with open(os.path.join(data_dir, 'sorted_cats.csv'), 'r') as f:
        read = [line.strip('\n').split('\t') for line in f.readlines()]
        colnames = read[0]
        cat_lines = read[1:]

    with open(os.path.join(data_dir, 'sorted_dogs.csv'), 'r') as f:
        read = [line.strip('\n').split('\t') for line in f.readlines()]
        dog_lines = read[1:]

    cat_lines = [line[9:-1] for line in cat_lines]
    dog_lines = [line[9:-1] for line in dog_lines]

    split_cat_lines = [[word.lower().translate(
                        str.maketrans('', '', string.punctuation)
                        ) for line in user for word in line.split()]
                        for user in cat_lines]
    split_dog_lines = [[word.lower().translate(
                        str.maketrans('', '', string.punctuation)
                        ) for line in user for word in line.split()]
                        for user in dog_lines]

    both_lines = split_cat_lines + split_dog_lines

    freq = {}
    idx_to_word = {}
    word_to_idx = {}
    num_words = 0
    for user in both_lines:
        for word in user:
            if word in freq.keys():
                freq[word] += 1
            else:
                freq[word] = 1
                word_to_idx[word] = num_words
                idx_to_word[num_words] = word
                num_words += 1

    cat_indexes = [[word_to_idx[word] for word in user] for user in split_cat_lines]
    dog_indexes = [[word_to_idx[word] for word in user] for user in split_dog_lines]

    new_cat_indexes = []
    max_threshold = 15000  # 15000
    min_threshold = 15  # 15
    kept_word_idxs = []
    for user in cat_indexes:
        new_user = []
        for idx in user:
            if min_threshold < freq[idx_to_word[idx]] <= max_threshold:
                new_user.append(idx)
                kept_word_idxs.append(idx)
        new_cat_indexes.append(new_user)

    new_dog_indexes = []
    for user in dog_indexes:
        new_user = []
        for idx in user:
            if min_threshold < freq[idx_to_word[idx]] <= max_threshold:
                new_user.append(idx)
                kept_word_idxs.append(idx)
        new_dog_indexes.append(new_user)

    adjusted_num_words = len(list(set(kept_word_idxs)))
    old_to_new_idx = {idx: i for (idx, i) in zip(list(set(kept_word_idxs)),
                                                 range(adjusted_num_words))}

    cat_words = [[idx_to_word[word] for word in user] for user in new_cat_indexes]
    dog_words = [[idx_to_word[word] for word in user] for user in new_dog_indexes]

    all_cat_embeds = torch.zeros(len(cat_lines), embed_dim).to(device)
    all_dog_embeds = torch.zeros(len(dog_lines), embed_dim).to(device)

    print("Loading fastText embedding model...")
    ft_embeds = ft.load_model(ft_dir)

    print("Preparing fastText embeddings...")
    for user_idx, user in tqdm(enumerate(cat_words)):
        all_word_embeds_for_user = torch.zeros(len(user), embed_dim)
        for word_idx, word in enumerate(user):
            word_embed = ft_embeds.get_word_vector(word)
            all_word_embeds_for_user[word_idx, :] = torch.tensor(word_embed)
        all_cat_embeds[user_idx, :] = torch.mean(all_word_embeds_for_user, dim=0)

    for user_idx, user in tqdm(enumerate(dog_words)):
        all_word_embeds_for_user = torch.zeros(len(user), embed_dim)
        for word_idx, word in enumerate(user):
            word_embed = ft_embeds.get_word_vector(word)
            all_word_embeds_for_user[word_idx, :] = torch.tensor(word_embed)
        all_dog_embeds[user_idx, :] = torch.mean(all_word_embeds_for_user, dim=0)

    print("Size of embeddings:", all_cat_embeds.element_size() * all_cat_embeds.nelement())
    print("Original number of words:", num_words)

    print("Saving embeddings...")
    torch.save(all_cat_embeds, os.path.join(data_dir, 'cat_embeds_ft.pt'))
    torch.save(all_dog_embeds, os.path.join(data_dir, 'dog_embeds_ft.pt'))


def prepare_pokec_bert_embeddings(rootdir):
    '''
    Input (from below directory): 10,000 users with 59 columns, one each for
                                  cats/dogs
    Output: torch tensor of dim [num_users X num_cols X bert_hidden_size], one
            each for cats/dogs

            num_users is 10,000, num_cols is 49 (10 columns removed with less
            relevant info), bert_hidden_size is 768, taking the output of the
            last layer and taking the average over all words

            cat_users, dog_users: indexes correspond to dim 0 indexes of above
            output tensors, to be used with adjacency matrix
    '''
    data_dir = os.path.join(rootdir, 'Pokec', 'raw')
    bert_dir = 'bert-base-multilingual-cased'

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')

    with open(os.path.join(data_dir, 'sorted_cats.csv'), 'r') as f:
        read = [line.strip('\n').split('\t') for line in f.readlines()]
        colnames = read[0]
        cat_lines = read[1:]

    with open(os.path.join(data_dir, 'sorted_dogs.csv'), 'r') as f:
        read = [line.strip('\n').split('\t') for line in f.readlines()]
        dog_lines = read[1:]

    cat_lines = [line[9:-1] for line in cat_lines]
    dog_lines = [line[9:-1] for line in dog_lines]
        # Starting from 'I_am_working_in-field' up to and not including 'more'

    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    model = BertModel.from_pretrained(bert_dir).to(device)

    print("Extracting BERT embeddings...")
    all_cat_embeds = torch.zeros(len(cat_lines), len(cat_lines[0]), model.config.hidden_size).to(device)
    all_dog_embeds = torch.zeros(len(dog_lines), len(dog_lines[0]), model.config.hidden_size).to(device)
        # num_total_examples, num_columns, hidden_dim

    with torch.no_grad():
        for ex_idx, (cat_user, dog_user) in tqdm(enumerate(zip(cat_lines, dog_lines))):
            for col_idx, (cat_col, dog_col) in enumerate(zip(cat_user, dog_user)):
                tokenized_cat_col = tokenizer(cat_col, return_tensors='pt',
                                              truncation=True).to(device)
                cat_embed_last_hidden_states = model(**tokenized_cat_col)[0]
                all_cat_embeds[ex_idx, col_idx, :] = torch.mean(cat_embed_last_hidden_states, dim=1)

                tokenized_dog_col = tokenizer(dog_col, return_tensors='pt',
                                              truncation=True).to(device)
                dog_embed_last_hidden_states = model(**tokenized_dog_col)[0]
                all_dog_embeds[ex_idx, col_idx, :] = torch.mean(dog_embed_last_hidden_states, dim=1)

    torch.save(all_cat_embeds, os.path.join(data_dir, 'cat_embeds_bert_avg_cased.pt'))
    torch.save(all_dog_embeds, os.path.join(data_dir, 'dog_embeds_bert_avg_cased.pt'))


def prepare_pokec_sentencebert_embeddings(rootdir):
    data_dir = os.path.join(rootdir, 'Pokec', 'raw')
    model_dir = 'distiluse-base-multilingual-cased-v2'
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')

    model = SentenceTransformer(model_dir)
    model = model.to(device)

    with open(os.path.join(data_dir, 'sorted_cats.csv'), 'r') as f:
        read = [line.strip('\n').split('\t') for line in f.readlines()]
        colnames = read[0]
        cat_lines = read[1:]

    with open(os.path.join(data_dir, 'sorted_dogs.csv'), 'r') as f:
        read = [line.strip('\n').split('\t') for line in f.readlines()]
        dog_lines = read[1:]

    cat_lines = [line[9:-1] for line in cat_lines]
    dog_lines = [line[9:-1] for line in dog_lines]
        # Starting from 'I_am_working_in-field' up to and not including 'more'

    print("Extracting SBERT embeddings...")
    embed_dim = model.get_sentence_embedding_dimension()
    all_cat_embeds = torch.zeros(len(cat_lines), len(cat_lines[0]), embed_dim).to(device)
    all_dog_embeds = torch.zeros(len(dog_lines), len(dog_lines[0]), embed_dim).to(device)
        # num_total_examples, num_columns, hidden_dim

    with torch.no_grad():
        for ex_idx, (cat_user, dog_user) in tqdm(enumerate(zip(cat_lines, dog_lines))):
            for col_idx, (cat_col, dog_col) in enumerate(zip(cat_user, dog_user)):
                all_cat_embeds[ex_idx, col_idx, :] = model.encode(cat_col,
                        convert_to_tensor=True)
                all_dog_embeds[ex_idx, col_idx, :] = model.encode(dog_col,
                        convert_to_tensor=True)

    torch.save(all_cat_embeds, os.path.join(data_dir, 'cat_embeds_sbert.pt'))
    torch.save(all_dog_embeds, os.path.join(data_dir, 'dog_embeds_sbert.pt'))


def prepare_pokec_graph(rootdir, feat_type='sbert', new_amat=True):
    data_dir = os.path.join(rootdir, 'Pokec')
    filename = 'soc-pokec-relationships.txt'
    raw_data_dir = os.path.join(data_dir, 'raw')

    file_path = os.path.join(raw_data_dir, filename)

    if feat_type == 'sbert':
        cat_embeds_file = 'cat_embeds_sbert.pt'
        dog_embeds_file = 'dog_embeds_sbert.pt'
    elif feat_type == 'bert_avg':
        cat_embeds_file = 'cat_embeds_bert_avg_cased.pt'
        dog_embeds_file = 'dog_embeds_bert_avg_cased.pt'
    elif feat_type == 'bows':
        cat_embeds_file = 'cat_embeds_bows.pt'
        dog_embeds_file = 'dog_embeds_bows.pt'
    elif feat_type == 'ft':
        cat_embeds_file = 'cat_embeds_ft.pt'
        dog_embeds_file = 'dog_embeds_ft.pt'
    else:
        raise Exception(f"{feat_type} not a valid feature type for "
                        "preparing pokec embeddings ('sbert' or 'bert_avg')")

    device = torch.device('cuda' if torch.cuda.is_available()
                               else 'cpu')

    print("Loading embeddings...")
    all_cat_embeds = torch.load(os.path.join(raw_data_dir, cat_embeds_file), map_location=device)
    all_dog_embeds = torch.load(os.path.join(raw_data_dir, dog_embeds_file), map_location=device)
    all_embeds = torch.cat((all_cat_embeds, all_dog_embeds), dim=0)
    del all_cat_embeds
    del all_dog_embeds
    ys = torch.tensor([0 for _ in range(10000)] + [1 for _ in range(10000)])

    cat_users = torch.load(os.path.join(raw_data_dir, 'cat_users.pt'), map_location=device)
    dog_users = torch.load(os.path.join(raw_data_dir, 'dog_users.pt'), map_location=device)

    if new_amat:
        subsampled_amat = prepare_subsampled_amat(file_path, cat_users,
                                                  dog_users, device)
    else:
        subsampled_amat = torch.load(os.path.join(raw_data_dir,
                                     'pokec-sub-amat.pt'), map_location=device)

    # Last preprocessing on all_embeds:
    if not feat_type in ['bows', 'ft']:
        all_embeds = simplify_pokec_node_feats(all_embeds)

    print("Preparing masks...")
    # Creating masks:
    train_ratio = 0.8
    val_ratio = 0.1
    num_train = all_embeds.shape[0] * train_ratio
    num_val = int(all_embeds.shape[0] * val_ratio)
    num_test = int(all_embeds.shape[0] - num_train - num_val)
    num_classes = ys.unique().shape[0]
    num_train_per_class = int(num_train / num_classes)

    train_mask = torch.zeros(all_embeds.shape[0], dtype=torch.bool, device=device)
    val_mask = torch.zeros(all_embeds.shape[0], dtype=torch.bool, device=device)
    test_mask = torch.zeros(all_embeds.shape[0], dtype=torch.bool, device=device)

    for class_val in range(num_classes):
        trn_idx = (ys == class_val).nonzero().view(-1)
        trn_idx = trn_idx[torch.randperm(trn_idx.shape[0])[:num_train_per_class]]
        train_mask[trn_idx] = True

    non_trains = (~train_mask).nonzero().view(-1)
    non_trains = non_trains[torch.randperm(non_trains.shape[0])]

    val_mask[non_trains[:num_val]] = True
    test_mask[non_trains[num_val:num_val+num_test]] = True

    print("Saving prepared graph...")
    data = Data(x=all_embeds, edge_index=subsampled_amat, y=ys,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    torch.save(data, os.path.join(data_dir, 'processed',
                                  f'pokec-pets_{feat_type}_cased.pt'))
    return data


def read_in_chunks(file_path):
    with open(file_path, 'r') as f:
        for line in f.readlines():
            yield line


def prepare_subsampled_amat(file_path, cat_users, dog_users, device):
    pokec_amat = np.empty((30622564, 2), dtype=np.int64)

    print("Loading adjacency matrix...")
    for idx, line in tqdm(enumerate(read_in_chunks(file_path))):
        pokec_amat[idx, :] = np.array(line.strip().split('\t'), dtype=np.int64)

    pokec_amat = torch.from_numpy(pokec_amat).t().to(device)

    # Subsampling graph based on whether user ids from cat_users and dog_users
    # are present or not in pokec_amat
    allowed_idxs = torch.cat((cat_users, dog_users))

    subsampled_amat = torch.zeros((2, 1), dtype=torch.int64, device=device)

    print("Subsampling adjacency matrix...")
    for idx in tqdm(range(pokec_amat.shape[1])):
        col = pokec_amat[:, idx]
        if col[0].item() in allowed_idxs and col[1].item() in allowed_idxs:
            subsampled_amat = torch.cat((subsampled_amat,
                                         col.unsqueeze(dim=1)), dim=1)

    del pokec_amat
    subsampled_amat = subsampled_amat[:, 1:]

    # Converting indexes of this subsampled adjacency matrix to match the new
    # 20,000 dimension
    print("Reindexing adjacency matrix...")
    old_to_new_idx_cats = {k.item(): v
                           for k, v in zip(cat_users,
                                           range(len(list(cat_users))))}
    old_to_new_idx_dogs = {k.item(): v
                           for k, v in zip(dog_users,
                                           range(10000,
                                                 10000+len(list(dog_users))))}
    old_to_new_idx = {**old_to_new_idx_cats, **old_to_new_idx_dogs}

    for key, val in old_to_new_idx.items():
        subsampled_amat[subsampled_amat == key] = val

    return subsampled_amat


def simplify_pokec_node_feats(pokec_bert_embeds):
    return pokec_bert_embeds.mean(dim=1)


def prepare_pokec_main(feat_type='sbert'):
    '''
    Puts together above pokec preprocessing functions.
    '''
    ss = Settings()
    rootdir = ss.args.root_dir
    preprocess_pokec_dataset_df(rootdir, output_stats=False)
    if feat_type == 'sbert':
        prepare_pokec_sentencebert_embeddings(rootdir)
    elif feat_type == 'bert_avg':
        prepare_pokec_bert_embeddings(rootdir, method='average')
    elif feat_type == 'bows':
        prepare_pokec_bows_embeddings()
    elif feat_type == 'ft':
        prepare_pokec_fasttext_embeddings()
    else:
        raise Exception(f"{feat_type} not a valid feature type for "
                        "preparing pokec embeddings ('sbert' or 'bert_avg')")

    data = prepare_pokec_graph(rootdir, feat_type=feat_type)

    return data


def normalize_cm(cm):
    row_sum = np.sum(cm, 1)
    normalized_cm = cm / row_sum[:, None]
    return normalized_cm


'''
Early stopping for the main network.
'''

class EarlyStopping(object):

    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


'''
Dataset classes.
'''
class RedditS(Dataset):
    def __init__(self, root_dir, rate=0.1, transform=None, pre_transform=None):
        super(RedditS, self).__init__(root_dir, rate=0.1)
        self.rate = rate

    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        return ['data_{}_{}.pt'.format(self.split_type, i) for i in range(len(self.data))]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        pass

    def process(self):
        # Won't run if data_{}_{}.pt files already exist
        print("Small reddit data not found, preparing...")
        data = make_small_reddit(rate=self.rate)
        return data

    def get(self, idx):
        data = torch.load(os.path.join(self.root_dir, 'RedditS',
                                       'processed', 'data.pt'))
        return data


