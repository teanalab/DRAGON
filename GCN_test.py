import csv
import os
import dgl

import pandas as pd
import torch
import json
from torch.utils.data import DataLoader
# from CERModel import CERModel
# from SubgraphPruningDataset import SubgraphPruningDataset
# from transformers import BertTokenizer
import argparse

from tqdm import tqdm

from QADataset import QADataset
from GCNRankerQKV_heads import GCNRanker
import json_stream
import utils


def read_json(json_file):
    with open(json_file, 'r') as f:
         data = json.load(f)
    return data

def create_graphs_from_scores(score_file):
    # with open(score_file, 'r') as f:
    #     scores_data = json.load(f)
    with open(score_file, 'r') as f:
        # read = f.read()
        # data = json_stream.load(f)
        data = json_stream.load(f, persistent=True)
        standard_data = json_stream.to_standard_types(data)

    graph_data = {}
    for element in standard_data:
        for qid, candidates in element.items():
            graph_data[qid] = {}
            for candidate, scores in candidates.items():
                # The first six scores are for the candidate itself
                candidate_scores = scores[:6]

                # Remaining scores are for the candidate's neighbors
                neighbor_scores = scores[6:]
                num_neighbors = len(neighbor_scores) // 6  # Each neighbor has 6 scores

                # Reshape the neighbor scores to be a matrix of shape (num_neighbors, 6)
                if num_neighbors > 0:
                    neighbor_features = torch.tensor(neighbor_scores).view(num_neighbors, 6)
                else:
                    neighbor_features = torch.empty((0, 6))  # Handle cases where there are no neighbors

                # Create a tensor for the candidate node's features
                candidate_features = torch.tensor([candidate_scores])  # Shape (1, 6)

                # Combine candidate and neighbor features into a single tensor
                all_features = torch.cat([candidate_features, neighbor_features], dim=0)  # Shape (1 + num_neighbors, 6)

                # Create a DGL graph for this candidate and its neighbors
                g = dgl.graph(([], []))  # Empty graph
                g.add_nodes(1 + num_neighbors)  # Add nodes: 1 candidate + num_neighbors

                # Add edges between candidate (node 0) and its neighbors
                if num_neighbors > 0:
                    src = [0] * num_neighbors  # Candidate node is connected to all neighbors
                    dst = list(range(1, num_neighbors + 1))  # Neighbor nodes start from 1
                    g.add_edges(src, dst)
                    g.add_edges(dst, src)  # Add bidirectional edges

                # g = dgl.add_self_loop(g)
                # Assign the feature tensor to the graph's nodes
                g.ndata['feat'] = all_features.float()

                # Save the graph in the dictionary with candidate as the key
                graph_data[qid][candidate] = g


    return graph_data

# Read triples from the tsv file
def read_triples(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t', header=0)
    triples = df.values.tolist()
    return triples

def prune_and_rank_candidates(model, dataloader, graph_data, device, top_k):
    """
    Prune and rank candidates for each question based on the pruning score.

    Args:
    - model: The trained model.
    - dataset: The dataset with candidates and question entities.
    - device: The device to run the model on.
    - top_k: Number of top candidates to select (default: 100).

    Returns:
    - A dictionary where keys are question_ids and values are lists of top_k candidates.
    """
    model.eval()  # Set the model to evaluation mode
    question_candidates = {}
    with torch.no_grad():
    # Iterate over the dataset
        for iteration, qid in tqdm(enumerate(dataloader), desc="Model training"):

            pos, neg = dataloader[qid]["p"], dataloader[qid]["n"]
            # if neg == 'nan':
            #     continue
            g_pos = graph_data[qid][pos]
            g_pos_feature = g_pos.ndata['feat']
            g_neg = [graph_data[qid][n] for n in neg if not isinstance(neg[0], float)]
            if g_neg is None:
                g_neg_feature = []
            else:
                g_neg_feature = [g_neg[i].ndata['feat'] for i in range(len(g_neg))]
            # optimizer.zero_grad()
            # losses = []
            # loss = 0
            # g_pos = [graph_data[q][p] for q, p in zip(qid, pos) if not p == 'nan']
            # g_pos_feature = [g_pos[i].ndata['feat'] for i in range(len(g_pos))]
            # g_neg = [graph_data[q][n] for q, n in zip(qid, neg) if not n == 'nan']
            # g_neg_feature = [g_neg[i].ndata['feat'] for i in range(len(g_neg))]



            with torch.no_grad():
                candid_scores_list = []
                pos_scores = model(g_pos.to(device), g_pos_feature.to(device)).mean(dim=0)
                candid_scores_list.append((pos, pos_scores, 1))
                for gn, gnf, neg_name in zip(g_neg, g_neg_feature, neg):
                    gn, gnf = gn.to(device), gnf.to(device)
                    neg_scores = model(gn, gnf).mean(dim=0)
                    candid_scores_list.append((neg_name, neg_scores, 0))
                # for gp, gn, gpf, gnf in zip(g_pos, g_neg, g_pos_feature, g_neg_feature):
                #     gp, gn, gpf, gnf = gp.to(device), gn.to(device), gpf.to(device), gnf.to(device)
                #     pos_scores = model(gp, gpf).mean(dim=0)
                #     neg_scores = model(gn, gnf).mean(dim=0)
                # score = model(input_ids, attention_mask, token_type_ids, Enc_Q, Enc_C, relevance)



            # Store the candidate and its score
            if qid not in question_candidates:
                question_candidates[qid] = []
            # if not pos == 'nan' and (pos[0], pos_scores, 1) not in question_candidates[qid]:
            #     question_candidates[qid].extend([(pos[0], pos_scores, 1)])

            question_candidates[qid].append(candid_scores_list)

            # if idx > 2:
            #     break
            # print(f'question_id is :{question_id} and question_candidates[question_id] is: {question_candidates[question_id]}\n')
        # Rank and prune the candidates for each question
    pruned_candidates = {}
    #TODO: save question_candidates in CSV file

    # for question_id, candidates in question_candidates.items():
    #     score_list = [score for score in candidates[0]]
    #     # File path to save the CSV
    output_file = 'scores.csv'

        # Open the CSV file for writing
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(['question_id', 'candidate', 'score'])

        # Iterate through each question_id and its candidates
        for question_id, candidates in question_candidates.items():
            # Generate the score_list as (candidate, score) for each candidate
            score_list = [score for score in candidates[0]]
            # Write each (candidate, score) pair along with the question_id to the CSV
            for candidate, score, _ in score_list:
                writer.writerow([question_id, candidate, score.item()])


    pruned_candidates_metrics = {}
    for question_id, candidates in question_candidates.items():
        print("start sorting")
        score_list = [score for score in candidates[0]]
        # Sort candidates based on the pruning score (descending order)
        candidates = sorted(score_list, key=lambda x: x[1].item(), reverse=True)

        # Select the top_k candidates
        # top_candidates = [candidate for candidate, score, relevance in candidates[:top_k]]
        #
        # Save the top candidates in the dictionary
        # pruned_candidates[question_id] = top_candidates

        # Select the top_k candidates
        top_candidates_metrics = [(candidate, score.item(), relevance) for candidate, score, relevance in candidates[:top_k]]

        # Save the top candidates in the dictionary
        pruned_candidates_metrics[question_id] = top_candidates_metrics

    return pruned_candidates_metrics

def calculate_mrr(pruned_candidates, top_k):
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
    - pruned_candidates: The dictionary of pruned candidates (output of prune_and_rank_candidates).
    - top_k: Number of top candidates to consider for metrics.

    Returns:
    - MRR: Mean Reciprocal Rank score.
    """
    total_reciprocal_rank = 0.0
    num_questions = len(pruned_candidates)

    for question_id, candidates in pruned_candidates.items():
        # print(f"Calculating MRR for question_id: {question_id}")
        # `candidates` is a list of tuples [(candidate, score, relevance), ...]
        for rank, (_, _, relevance) in enumerate(candidates[:top_k]):
            if relevance == 1:
                # Calculate reciprocal rank as 1 / (rank + 1)
                total_reciprocal_rank += 1.0 / (rank + 1)
                break  # Stop after the first relevant item

    # Calculate MRR as the average of the reciprocal ranks
    mrr = total_reciprocal_rank / num_questions
    return total_reciprocal_rank, mrr

def calculate_metrics(pruned_candidates, top_k):
    """
    Calculate Mean Average Precision (MAP), Hit@1, and Hit@10.

    Args:
    - pruned_candidates: The dictionary of pruned candidates (output of prune_and_rank_candidates).
    - top_k: Number of top candidates to consider for metrics.

    Returns:
    - A dictionary with MAP, Hit@1, and Hit@10 scores.
    """
    total_map = 0
    total_hits_1 = 0
    total_hits_10 = 0
    total_hits_100 = 0
    num_questions = len(pruned_candidates)

    for question_id, candidates in pruned_candidates.items():
        print("start calculation metrics")
        # Sort candidates by their score (already sorted in prune_and_rank_candidates)
        relevant_indices = [i for i, (_, _, relevance) in enumerate(candidates) if relevance == 1]

        # Calculate Average Precision (AP) for this question
        if len(relevant_indices) > 0:
            ap = 0
            num_relevant = 0
            for i, idx in enumerate(relevant_indices):
                if idx < top_k:
                    num_relevant += 1
                    precision_at_i = num_relevant / (idx + 1)
                    ap += precision_at_i
            ap /= len(relevant_indices)
            total_map += ap

        # Check if relevant entity is at the first rank (Hit@1)
        if relevant_indices and relevant_indices[0] == 0:
            # print(relevant_indices[0])
            total_hits_1 += 1

        # Check if relevant entity is within top 10 (Hit@10)
        if relevant_indices and relevant_indices[0] < 10:
            total_hits_10 += 1

        if relevant_indices and relevant_indices[0] < 100:
            total_hits_100 += 1

    # Calculate final metrics
    mean_ap = total_map / num_questions
    hit_1 = total_hits_1
    hit_10 = total_hits_10
    hit_100 = total_hits_100

    # hit_1 = total_hits_1 / num_questions
    # hit_10 = total_hits_10 / num_questions



    return {
        "MAP": mean_ap,
        "Hit@1": hit_1,
        "Hit@10": hit_10,
        "Hit@100": hit_100
    }



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test', choices=['test'])

    parser.add_argument('--gpu', type=int, default=0, help="GPU device ID. Use -1 for CPU")
    parser.add_argument('--model_path', type=str,  help="Path to the saved model",
                        default= "/home/hi9115/GCN-IR/Rank_candidate_project_Option1/GCN-model/GCN-model/types-models/weighted-jaccard/train_model_loss_0.2055_epoch_20.pt")
    # parser.add_argument('--dataset_tsv', type=str,  help="Path to the TSV file with candidates",
    #                     default= "/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/reduced_candid_neighbors/relevance_train.tsv")
    # parser.add_argument('--score_file_dir',
    #                     default='/data/hi9115/Scores-Colbert/',
    #                     help="Directory containing score files")
    parser.add_argument('--score_file_dir',
                        default='/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/Scores-weighted-jaccard-types/',
                        help="Directory containing score files")
    parser.add_argument('--triple_file_dir',
                        default='/home/hi9115/GCN-IR/Rank_candidate_project_Option1/GCN-model/GCN-model/triples/',
                        help="Directory containing triple files")

    parser.add_argument('--top_k', type=int, help="number of candidates for pruning",
                        default = 100)
    args = parser.parse_args()

    # Setup device
    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    # Hyperparameters
    in_feats = 12
    hidden_size = 16
    out_feats = 12
    learning_rate = 2.5e-5
    epochs = 200

    # Load the trained model
    model = GCNRanker(in_feats, hidden_size, out_feats)
    model = model.to(device)
    checkpoint = torch.load(args.model_path)
    print(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    split = args.split
    # qblink = utils.load_qblink_split(split)
    score_file_dir = args.score_file_dir
    score_file = os.path.join(score_file_dir, f'scores_{split}.json')

    # Create graph data from scores file
    graph_data = create_graphs_from_scores(score_file)
    # Read triples for pairwise ranking
    triple_file_dir = args.triple_file_dir
    triples_file = os.path.join(triple_file_dir, f'dict_{split}.json')
    triples = read_triples(triples_file)

    # Create a dataset and dataloader
    # dataset = QADataset(triples)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    top_k = args.top_k


    triples = read_json(triples_file)


    # Prune and rank candidates for each question

    pruned_candidates_metrics = prune_and_rank_candidates(model, triples, graph_data, device, top_k=top_k)
    output_file2 ='/home/hi9115/GCN-IR/Rank_candidate_project_Option1/GCN-model/GCN-model/sorted.csv'
    with open(output_file2, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(['question_id', 'candidate', 'score'])

        # Iterate through each question_id and its candidates
        for question_id, candidates in pruned_candidates_metrics.items():
            # Generate the score_list as (candidate, score) for each candidate
            # score_list = [score for score in candidates[0]]
            # # Write each (candidate, score) pair along with the question_id to the CSV
            # for candidate, score in score_list:
            for candidate, score, relevance in candidates:
                writer.writerow([question_id, candidate, score, relevance])

    metrics = calculate_metrics(pruned_candidates_metrics, top_k=top_k)
    print(f"MAP: {metrics['MAP']:.4f}, Hit@1: {metrics['Hit@1']:.4f}, Hit@10: {metrics['Hit@10']:.4f}, Hit@100: {metrics['Hit@100']:.4f}")
    mrr_pure, mrr_divide_num_q = calculate_mrr(pruned_candidates_metrics, top_k)
    print(f"MRR is: {mrr_pure}, mrr_divide_num_q:{mrr_divide_num_q}")
    # Save the pruned candidates to a JSON file
    # save_pruned_candidates(pruned_candidates, args.output_file)

    # print(f"Pruned candidates saved to {args.output_file}")
