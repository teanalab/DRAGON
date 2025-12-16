import argparse

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch.nn as nn

from SubgraphPruningDataset import SubgraphPruningDataset
from CERModel import CERModel
from sklearn.metrics import average_precision_score
import operator


# Function to compute Mean Average Precision (MAP)
def calculate_map(predictions, labels):
    # Calculate Average Precision (AP) for each query
    ap_scores = []
    for pred, label in zip(predictions, labels):
        ap = average_precision_score(label, pred)
        ap_scores.append(ap)

    # Return Mean Average Precision (MAP)
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0

# Evaluation function for the test dataset
def evaluate_model(model, test_loader, loss_fn):
    model.eval()  # Set model to evaluation mode
    test_epoch_loss = 0.0
    test_samples = len(test_loader.dataset)

    # Store predictions and labels for each query (grouped by query)
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for indx in range(0, test_samples):
            sample = test_loader.dataset.__getitem__(indx)
            input_ids = sample['input_ids'].to(device)
            attention_mask = sample['attention_mask'].to(device)
            token_type_ids = sample['token_type_ids'].to(device)
            question_id = sample['question_id']
            candidate = sample['candidate']

            Enc_Q = sample['Enc_Q']
            Enc_C = sample['Enc_C']

            relevance = sample['relevance'].to(device)

            # Forward pass without backprop
            pruning_score, pruning_score_prob = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0),
                                  token_type_ids, Enc_Q, Enc_C, relevance)
            loss = loss_fn(pruning_score_prob.squeeze(), relevance)
            test_epoch_loss += loss.item()

            # Collect predictions and labels for MAP calculation
            all_predictions.append(pruning_score_prob.cpu().numpy())
            all_labels.append(relevance.cpu().numpy())

    # Calculate average test loss
    test_epoch_loss = test_epoch_loss / test_samples

    # Convert predictions and labels to the right format for MAP
    # `all_predictions` and `all_labels` should be lists of lists (one list per query)
    all_predictions = [pred.squeeze().tolist() for pred in all_predictions]
    all_labels = [label.squeeze().tolist() for label in all_labels]
    # ranked_entities = sorted(entity_scores, key=operator.itemgetter(1), reverse=True)

    # Compute Mean Average Precision (MAP)
    # map_score = average_precision_score(all_labels, torch.where(torch.tensor(all_predictions) == max(all_predictions), 1, 0))

    # map_score = av(all_predictions, all_labels)
    print(f"Test Loss: {test_epoch_loss:.4f}, MAP: {map_score:.4f}")
    return test_epoch_loss, map_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', default='/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/reduced_candid_neighbors/models/fine_tune_epoch2_lambda05_maxlen5120.pt', help="Path to the trained model.")
    # Define device and load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    # Initialize the model and move it to the appropriate device
    model = CERModel()
    model = model.to(device)

    # Load the saved model (best performing model)
    checkpoint = torch.load(args.modelpath)
    print('Epoch:', checkpoint['epoch'])
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load the test dataset
    test_dataset = SubgraphPruningDataset(
        '/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/reduced_candid_neighbors/relevance_test.tsv',
        '/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/entities.json',
        tokenizer, device
    )

    # DataLoader for batching test data
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Define the loss function (Mean Squared Error)
    loss_fn = nn.MSELoss()

    # Evaluate the model on the test set
    test_loss, map_score = evaluate_model(model, test_loader, loss_fn)
    print(f"Final Test Loss: {test_loss:.4f}, MAP: {map_score:.4f}")

