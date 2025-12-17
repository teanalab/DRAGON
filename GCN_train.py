import argparse
import csv
import os

import json_stream
from tqdm import tqdm

import utils

from json_stream.dump import JSONStreamEncoder
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
# from dgl.nn import GraphConv
from torch.utils.data import Dataset, DataLoader
from QADataset import QADataset
from GCNRankerQKV_heads import (GCNRanker)
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from GCN_test_new import calculate_mrr
import random
import numpy as np
import torch
import time
import random
import torch
import dgl
from tqdm import tqdm
from json_stream import load, to_standard_types

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
    total_mrr_100 = 0
    total_recall1 = 0.0
    total_recall10 = 0.0
    num_questions = len(pruned_candidates)

    for question_id, candidates in pruned_candidates.items():
        # print("start calculation metrics")
        # Sort candidates by their score (already sorted in prune_and_rank_candidates)
        relevant_indices = [i for i, (_, _, relevance) in enumerate(candidates[0]) if relevance == 1]
        num_relevant_total = len(relevant_indices)

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
            # For MRR@100, consider the first relevant candidate if it appears within the top_k
            first_relevant_rank = relevant_indices[0] + 1  # convert to 1-indexed rank
            if first_relevant_rank <= top_k:
                total_mrr_100 += 1.0 / first_relevant_rank
            else:
                total_mrr_100 += 0

        # Check if relevant entity is at the first rank (Hit@1)
        if relevant_indices and relevant_indices[0] == 0:
            # print(relevant_indices[0])
            total_hits_1 += 1

        # Check if relevant entity is within top 10 (Hit@10)
        if relevant_indices and relevant_indices[0] < 10:
            total_hits_10 += 1

        if relevant_indices and relevant_indices[0] < 100:
            total_hits_100 += 1

            # Recall@K: Count the number of relevant candidates retrieved in top K, then divide by total relevant.
            # Here we compute Recall@1 and Recall@10.
        retrieved_relevant_at_1 = sum(
            1 for i in range(min(1, len(candidates[0]))) if candidates[0][i][2] == 1)
        retrieved_relevant_at_10 = sum(
            1 for i in range(min(10, len(candidates[0]))) if candidates[0][i][2] == 1)
        if num_relevant_total > 0:
            recall1 = retrieved_relevant_at_1 / num_relevant_total
            recall10 = retrieved_relevant_at_10 / num_relevant_total
        else:
            recall1 = 0.0
            recall10 = 0.0

        total_recall1 += recall1
        total_recall10 += recall10
    # Calculate final metrics
    mean_ap = total_map / num_questions
    hit_1 = total_hits_1
    hit_10 = total_hits_10
    hit_100 = total_hits_100
    mrr_100 = total_mrr_100 / num_questions
    recall_1 = total_recall1 / num_questions
    recall_10 = total_recall10 / num_questions
    # hit_1 = total_hits_1 / num_questions
    # hit_10 = total_hits_10 / num_questions



    return {
        "MAP": mean_ap,
        "Hit@1": hit_1,
        "Hit@10": hit_10,
        "Hit@100": hit_100,
        "MRR@100": mrr_100,
        "Recall@1": recall_1,
        "Recall@10": recall_10

    }

def create_graphs_from_scores_test(score_file):
    # with open(score_file, 'r') as f:
    #     scores_data = json.load(f)
    max_neighbors = 1000  # Set the maximum number of neighbors
    start_time = time.time()
    with open(score_file, 'r') as f:
        # read = f.read()
        # data = json_stream.load(f)
        data = json_stream.load(f, persistent=True)
        standard_data = json_stream.to_standard_types(data)
    end_time = time.time()
    load_time = end_time - start_time
    print(f"Time taken to load the file: {load_time} seconds")
    graph_data = {}
    for element in tqdm(standard_data, desc="Creating Graph."):
        for qid, candidates in element.items():
            graph_data[qid] = {}
            for candidate, scores in candidates.items():
                # The first six scores are for the candidate itself
                candidate_scores = scores[:12]

                # Remaining scores are for the candidate's neighbors
                neighbor_scores = scores[12:]
                num_neighbors = len(neighbor_scores) // 12  # Each neighbor has 6 scores

                # Limit the number of neighbors to `max_neighbors`
                if num_neighbors > max_neighbors:
                    sampled_indices = random.sample(range(num_neighbors), max_neighbors)
                    neighbor_scores = [
                        neighbor_scores[i * 12:(i + 1) * 12] for i in sampled_indices
                    ]
                    neighbor_scores = [item for sublist in neighbor_scores for item in sublist]
                    num_neighbors = max_neighbors

                # Reshape the neighbor scores to be a matrix of shape (num_neighbors, 6)
                if num_neighbors > 0:
                    neighbor_features = torch.tensor(neighbor_scores).view(num_neighbors, 12)
                else:
                    neighbor_features = torch.empty((0, 12))  # Handle cases where there are no neighbors

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

                # Add a self-loop for the candidate node only
                g = dgl.add_self_loop(g)   # Self-loop for candidate node only
                # Assign the feature tensor to the graph's nodes
                g.ndata['feat'] = all_features.float()

                # Save the graph in the dictionary with candidate as the key
                graph_data[qid][candidate] = g


    return graph_data


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
    total_loss_test = 0
    with torch.no_grad():
    # Iterate over the dataset
        for iteration, qid in tqdm(enumerate(dataloader), desc="Model eval"):
            pos, neg = dataloader[qid]["p"], dataloader[qid]["n"]
            g_pos = graph_data[qid][pos]
            g_pos_feature = g_pos.ndata['feat']
            g_neg = [graph_data[qid][n] for n in neg if not isinstance(neg[0], float)]
            if g_neg is None:
                g_neg_feature = []
            else:
                g_neg_feature = [g_neg[i].ndata['feat'] for i in range(len(g_neg))]

            with torch.no_grad():
                candid_scores_list = []
                pos_scores = model(g_pos.to(device), g_pos_feature.to(device)).mean(dim=0)
                candid_scores_list.append((pos, pos_scores, 1))
                for gn, gnf, neg_name in zip(g_neg, g_neg_feature, neg):
                    gn, gnf = gn.to(device), gnf.to(device)
                    neg_scores = model(gn, gnf).mean(dim=0)
                    candid_scores_list.append((neg_name, neg_scores, 0))

                loss = listnet_loss_with_penalty(pos_scores, neg_scores)
                total_loss_test += loss.item()
            # Store the candidate and its score
            if qid not in question_candidates:
                question_candidates[qid] = []


            candidates = sorted(candid_scores_list, key=lambda x: x[1].item(), reverse=True)
            # top_candidates_metrics = [(candidate, score.item(), relevance) for candidate, score, relevance in
            #                           candidates[:top_k]]
            top_candidates_metrics = [(candidate, score.item(), relevance) for candidate, score, relevance in
                                      candidates]
            question_candidates[qid].append(top_candidates_metrics)
    
    return question_candidates, total_loss_test, iteration


def create_graphs_from_scores(score_file):
    print("Inside create graph file")
    graph_data = {}

    max_neighbors = 1000  # Set the maximum number of neighbors

    with open(score_file, 'r') as f:
        data = load(f, persistent=True)
        standard_data = to_standard_types(data)

        for element in tqdm(standard_data, desc="Creating Graph."):
            for qid, candidates in element.items():
                graph_data[qid] = {}
                for candidate, scores in candidates.items():
                    # The first six scores are for the candidate itself
                    candidate_scores = scores[:12]

                    # Remaining scores are for the candidate's neighbors
                    neighbor_scores = scores[12:]
                    num_neighbors = len(neighbor_scores) // 12  # Each neighbor has 6 scores

                    # Limit the number of neighbors to `max_neighbors`
                    if num_neighbors > max_neighbors:
                        sampled_indices = random.sample(range(num_neighbors), max_neighbors)
                        neighbor_scores = [
                            neighbor_scores[i * 12:(i + 1) * 12] for i in sampled_indices
                        ]
                        neighbor_scores = [item for sublist in neighbor_scores for item in sublist]
                        num_neighbors = max_neighbors

                    # Reshape the neighbor scores to be a matrix of shape (num_neighbors, 6)
                    if num_neighbors > 0:
                        neighbor_features = torch.tensor(neighbor_scores).view(num_neighbors, 12)
                    else:
                        neighbor_features = torch.empty((0, 12))  # Handle cases where there are no neighbors

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
                    g = dgl.add_self_loop(g)  # Self-loop for candidate node only
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


def merge_json_files(file1, file2):
    """
    Merge two JSON files into one dictionary.

    Args:
        file1: Path to the first JSON file.
        file2: Path to the second JSON file.

    Returns:
        Merged dictionary containing data from both files.
    """
    # Read the first JSON file
    with open(file1, 'r') as f1:
        data1 = json.load(f1)

    # Read the second JSON file
    with open(file2, 'r') as f2:
        data2 = json.load(f2)

    # Merge the dictionaries
    merged_data = {**data1, **data2}  # Combines data1 and data2, data2 overwrites keys in data1 if duplicates exist
    with open("Path to the merged dictionary file", 'w') as f_out:
        json.dump(merged_data, f_out, indent=4)
    return merged_data

def read_json(json_file):
    with open(json_file, 'r') as f:
         data = json.load(f)
    return data

# Pairwise hinge loss function
def pairwise_loss(pos_scores, neg_scores, margin=1.0):
    loss = (margin - pos_scores + neg_scores).mean()
    assert loss >= 0, "Loss should be non-negative"
    return F.relu(loss)


def listnet_loss_with_penalty(pos_scores, neg_scores):
    """
    Compute a modified ListNet loss with penalties based on the rank of the golden answer.

    Args:
    - pos_scores: Tensor of positive candidate scores (1-dimensional).
    - neg_scores: Tensor of scores for negative candidates (1-dimensional).
    - device: The device (e.g., "cuda" or "cpu") to run the computations on.

    Returns:
    - Loss: Computed modified ListNet loss.
    """
    # Combine positive and negative scores into a single tensor
    pos_scores = pos_scores.unsqueeze(0) if pos_scores.dim() == 1 else pos_scores
    all_scores = torch.cat((pos_scores.view(-1, 1), neg_scores.view(-1, 1)), dim=0)

    # Create ground-truth relevance distribution
    relevance = torch.zeros_like(all_scores, requires_grad=False).to(device)
    relevance[0] = 1.  # Golden answer is relevant

    # Apply softmax to compute predicted distribution
    # predicted_distribution = F.softmax(all_scores, dim=0)
    predicted_distribution = F.softmax(all_scores, dim=0)
    # predicted_distribution = all_scores


    # True distribution for the golden answer
    true_distribution = relevance.float()

    # Compute rank of the golden answer (position where it ranks in all scores)
    sorted_indices = torch.argsort(all_scores.squeeze(), descending=True)
    golden_rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1  # +1 for 1-based rank

    # Weight using MAE: Higher penalty for farther ranks
    weight = torch.abs(torch.arange(1, len(all_scores) + 1, device=device) - 1).float()  # 0 for rank@1
    rank_penalty = weight[golden_rank - 1] * 10  # Extract penalty for golden_rank

    # Compute KL-divergence
    kl_loss = F.kl_div(predicted_distribution, true_distribution, log_target=True)
    l1_loss_fn = nn.L1Loss()
    mae = l1_loss_fn(predicted_distribution, true_distribution)
    ce = nn.CrossEntropyLoss()
    ce_loss = ce(all_scores, relevance.float())
    kl_loss = kl_loss * rank_penalty

    return kl_loss

def listnet_loss(pos_scores, neg_scores):
    """
    Compute the ListNet loss using KL divergence.

    Args:
    - pos_scores: Tensor of positive candidate score (1-dimensional).
    - neg_scores: Tensor of scores for negative candidates (1-dimensional).

    Returns:
    - Loss: Computed ListNet loss.
    """
    # Combine positive and negative scores into a single tensor
    # Ensure pos_scores is a 2D tensor
    pos_scores = pos_scores.unsqueeze(0) if pos_scores.dim() == 1 else pos_scores

    all_scores = torch.cat((pos_scores.view(-1,1), neg_scores.view(-1,1)), dim=0)  # Concatenate along rows

    relevance = torch.zeros_like(all_scores, requires_grad=False).to(device)
    relevance[0] = 1.  # Only the positive sample is relevant

    # Apply softmax to get distributions
    predicted_distribution = F.log_softmax(all_scores, dim=0)

    true_distribution = F.softmax(relevance.float(), dim=0)

    sorted_indices = torch.argsort(all_scores.squeeze(), descending=True)
    golden_rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1  # +1 for 1-based rank
    # Weight using MAE: Higher penalty for farther ranks
    weight = torch.abs(torch.arange(1, len(all_scores) + 1, device=device) - 1).float()  # 0 for rank@1
    rank_penalty = weight[golden_rank - 1]  # Extract penalty for golden_rank

    # Compute KL-divergence between the true and predicted distributions
    loss = F.kl_div(predicted_distribution, true_distribution)

    bce = F.binary_cross_entropy(all_scores, relevance.float())
    weighted_bce = bce * rank_penalty
    return loss, weighted_bce

def write_gradients_to_tensorboard(model, writer, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f"Gradients/{name}", param.grad, step)

def eval_model(model,triples_test, graph_data, device, top_k=100 ):
    # Prune and rank candidates for each question
    pruned_candidates_metrics, total_loss_test, num_questions = prune_and_rank_candidates(model, triples_test, graph_data, device, top_k=top_k)
    output_file2 = '/home/hi9115/GCN-IR/Rank_candidate_project_Option1/GCN-model/GCN-model/sorted.csv'
    with open(output_file2, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(['question_id', 'candidate', 'score'])

        # Iterate through each question_id and its candidates
        for question_id, candidates in pruned_candidates_metrics.items():
            
            for candidate, score, relevance in candidates[0]:
                writer.writerow([question_id, candidate, score, relevance])

    metrics = calculate_metrics(pruned_candidates_metrics, top_k=top_k)
    print(
        f"MAP: {metrics['MAP']:.4f}, Hit@1: {metrics['Hit@1']:.4f}, Hit@10: {metrics['Hit@10']:.4f}, Hit@100: {metrics['Hit@100']:.4f}, MRR@100: {metrics['MRR@100']:.4f}, Recall@1: {metrics['Recall@1']:.4f}, Recall@10: {metrics['Recall@10']:.4f}")
    
    return total_loss_test, num_questions


# Training function
def train(model, train_triples,val_triples, triples_test, graph_data_train, graph_data_dev, graph_data_test, optimizer, device, model_path,experiments_path , start_epoch, epochs=10):
    # continue_var = 0
    # loss_records = []
    writer = SummaryWriter(log_dir=experiments_path)
    score_ranges = []

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-8)

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        iteration = 0
        for iteration, qid in tqdm(enumerate(train_triples), desc=f"Iteration for epoch: {epoch}"):
            pos, neg = graph_data_train[qid]["p"], graph_data_train[qid]["n"]
            if isinstance(neg[0], float):
                continue
            g_pos = graph_data_train[qid][pos]
            g_pos_feature = g_pos.ndata['feat']
            g_neg = [graph_data_train[qid][n] for n in neg if not isinstance(n, float)]
            g_neg_feature = [g_neg[i].ndata['feat'] for i in range(len(g_neg))]

            pos_scores = model(g_pos.to(device), g_pos_feature.to(device)).mean(dim=0)

            neg_scores_list = []
            for gn, gnf in zip(g_neg, g_neg_feature):
                gn, gnf = gn.to(device), gnf.to(device)
                # neg_scores = model(gn, gnf).squeeze(0)  # 1,1 -> 1

                neg_scores = model(gn, gnf).mean(dim=0)  # 1,1 -> 1
                # print("neg_scores:", neg_scores)  # Debugging line
                neg_scores_list.append(neg_scores)
            if len(neg_scores_list) > 0:
                neg_scores_tensor = torch.stack(neg_scores_list)  # Convert list to tensor
                neg_scores_tensor_backup = neg_scores_tensor.squeeze(-1)
                top_neg_scores, _ = torch.topk(neg_scores_tensor_backup.view(1,-1), k=min(13, len(neg_scores_tensor_backup)),
                                                         largest=True)
            else:
                print(f"No valid negative samples for query {qid}. Skipping.")
                continue

            # Debugging: Save the range of positive and negative scores
            pos_range = (pos_scores.min().item(), pos_scores.max().item())
            neg_range = (neg_scores_tensor.min().item(), neg_scores_tensor.max().item())
            score_ranges.append({
                "Epoch": epoch + 1,
                "Iteration": iteration + 1,
                "Pos_Scores_Min": pos_range[0],
                "Pos_Scores_Max": pos_range[1],
                "Neg_Scores_Min": neg_range[0],
                "Neg_Scores_Max": neg_range[1],
            })
           
            loss_kl = listnet_loss_with_penalty(pos_scores, top_neg_scores)
           
            loss = loss_kl

            optimizer.zero_grad()
            loss.backward()

            # Write gradients to TensorBoard
            write_gradients_to_tensorboard(model, writer, epoch * len(dataloader) + iteration)

            optimizer.step()
            scheduler.step(epoch + iteration / len(dataloader))  # Update learning rate
            writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)

            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/iteration}')
        epoch_model_path = f"{model_path}_loss_{total_loss/iteration:.4f}_epoch_{epoch + 1}.pt"
        #TODO: save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_epoch': total_loss
        }, epoch_model_path)
        # Write total loss and loss records to TensorBoard
        writer.add_scalar('Total_Loss_train', total_loss, epoch)
        

        # --------------------------------------------------
        # VALIDATION (NO GRADIENTS)
        # --------------------------------------------------
        model.eval()
        val_loss = 0
        val_count = 0

        with torch.no_grad():
            for qid in val_triples:
                pos, neg = val_triples[qid]["p"], val_triples[qid]["n"]
                if isinstance(neg[0], float):
                    continue

                g_pos = graph_data[qid][pos]
                g_pos_feature = g_pos.ndata['feat']

                g_neg = [graph_data_dev[qid][n] for n in neg if not isinstance(n, float)]
                g_neg_feature = [g.ndata['feat'] for g in g_neg]

                pos_scores = model(
                    g_pos.to(device),
                    g_pos_feature.to(device)
                ).mean(dim=0)

                neg_scores_list = []
                for gn, gnf in zip(g_neg, g_neg_feature):
                    neg_scores = model(
                        gn.to(device),
                        gnf.to(device)
                    ).mean(dim=0)
                    neg_scores_list.append(neg_scores)

                if len(neg_scores_list) == 0:
                    continue

                neg_scores_tensor = torch.stack(neg_scores_list).squeeze(-1)
                top_neg_scores, _ = torch.topk(
                    neg_scores_tensor.view(1, -1),
                    k=min(13, neg_scores_tensor.numel()),
                    largest=True
                )

                loss = listnet_loss_with_penalty(pos_scores, top_neg_scores)
                val_loss += loss.item()
                val_count += 1

        avg_val_loss = val_loss / max(1, val_count)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --------------------------------------------------
        # EARLY STOPPING
        # --------------------------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }, f"{model_path}_best.pt")

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        total_loss_test, num_questions_test = eval_model(model,triples_test, graph_data_test, device)
        print(f'Epoch {epoch+1}/{epochs}, loss_test: {total_loss_test/num_questions_test}')
        writer.add_scalar('total_loss_test', total_loss_test, epoch)


    writer.flush()



    # Save score ranges to Excel
    score_df = pd.DataFrame(score_ranges)
    score_df.to_excel(f"{model_path}_score_ranges.xlsx", index=False)
    print(f"Score ranges saved to {model_path}_score_ranges.xlsx")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train', choices=['train', 'dev', 'test'])
    parser.add_argument('--gpu', type=int, default= 0, help="GPU device ID. Use -1 for CPU training")
    
    parser.add_argument('--score_file_dir',
                        help="Directory containing score files")
    parser.add_argument('--triple_file_dir',
                        help="Directory containing triple files")
    parser.add_argument('--model_path', type = str,
                         help="Path to save the model")
    parser.add_argument('--experiments_path', type = str,
                        help="Path to save reports")

    parser.add_argument('--resume_from', type=str, help="Path to resume from")


    args = parser.parse_args()
    print(args)

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')


    # Hyperparameters
    in_feats = 12
    # hidden_size = 32
    hidden_size = 24
    out_feats = 12
    learning_rate = 2.5e-5
    epochs = 200
    split = args.split
    score_file_dir = args.score_file_dir

    score_file = os.path.join(score_file_dir, f'scores_{split}.json')
    score_file_dev = os.path.join(score_file_dir, f'scores_dev.json')

    # Create graph data from scores file
    print("before graph data")
    # For training data
    graph_data_train = create_graphs_from_scores(score_file)
    graph_data = create_graphs_from_scores(score_file_dev)
    # Read triples for pairwise ranking
    triple_file_dir = args.triple_file_dir
    triples_file_train = os.path.join(triple_file_dir, f'dict_{split}.json')
    triples_file_dev = os.path.join(triple_file_dir, f'dict_dev.json')

    # Merge the two JSON files
    # merged_triples = merge_json_files(triples_file, triples_file_dev)

    # Define the model and optimizer
    model = GCNRanker(in_feats, hidden_size, out_feats)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    start_epoch = 0  # Default start at epoch 0

    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        print(f"Resumed training from epoch {start_epoch}")
        print(f"Optimizer is {optimizer}")

    # create test graph
    score_file_test = os.path.join(score_file_dir, f'scores_test.json')
    # Create graph data from scores file
    graph_data_test = create_graphs_from_scores_test(score_file_test)
    # Read triples for pairwise ranking
    triple_file_dir = args.triple_file_dir
    triples_file_test = os.path.join(triple_file_dir, f'dict_test.json')
    triples_test = read_json(triples_file_test)


    # Continue training
    train(model, train_triples=triples_file_train, val_triples=triples_file_dev, test_triples=triples_test, graph_data_train=graph_data_train,graph_data_dev=graph_data_dev, graph_data_test=graph_data_test, optimizer=optimizer, device=device, model_path=args.model_path, experiments_path=args.experiments_path, start_epoch=start_epoch, epochs=epochs)