import torch
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from transformers import AdamW, BertTokenizer
import torch.nn as nn

import utils
from SubgraphPruningDataset import SubgraphPruningDataset
from CERModel import CERModel
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings("ignore")


# Training loop
def train_model(model, train_loader, dev_loader, optimizer, loss_fn, save_model_path, num_epochs=5):
    best_dev_loss = float('inf')
    best_epoch = -1
    lamda = 0.5
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        train_epoch_loss = 0.0
        epoch_start_time = time.time()

        train_samples = len(train_loader.dataset)
        pbar = tqdm(train_samples)

        for indx in range(train_samples):
            sample = train_loader.dataset.__getitem__(indx)
            pbar.update(1)

            input_ids = sample['input_ids'].to(device)
            attention_mask = sample['attention_mask'].to(device)
            token_type_ids = sample['token_type_ids'].to(device)

            Enc_Q = sample['Enc_Q']

            Enc_C = sample['Enc_C']

            relevance = sample['relevance'].to(device)  # Reshape for MSE

            # Forward pass through the model
            optimizer.zero_grad()
            cosine_sim, score = model(input_ids.unsqueeze(0),
                                                      attention_mask.unsqueeze(0), token_type_ids,
                                                      Enc_Q, Enc_C, relevance
                                                      )


            # Compute loss

            if relevance == 1:
                alpha = 1
            else:
                alpha = -1
            prob_pruning_score = lamda * cosine_sim + (1 - lamda) * score
            gt_pruning_score = lamda * cosine_sim + (1 - lamda) * alpha
            loss = loss_fn(gt_pruning_score, prob_pruning_score)
            # if relevance == 0:
            #     scale = 0.1
            # else:
            #     scale = 1e2
            # train_epoch_loss += loss.item() * scale
            train_epoch_loss += loss.item()


            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        train_epoch_loss = train_epoch_loss / train_samples
        epoch_model_path = f"{save_model_path}_loss_{train_epoch_loss:.4f}_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_epoch': train_epoch_loss
        }, epoch_model_path)

        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_epoch_loss / len(train_loader)}")
        # dev_epoch_loss = 0.0
        # model.eval()
        # dev_samples = len(dev_loader.dataset)
        # with torch.no_grad():
        #     for indx in range(dev_samples):
        #         sample = dev_loader.dataset.__getitem__(indx)
        #         input_ids = sample['input_ids'].to(device)
        #         attention_mask = sample['attention_mask'].to(device)
        #         token_type_ids = sample['token_type_ids'].to(device)
        #         Enc_Q = sample['Enc_Q']
        #
        #         Enc_C = sample['Enc_C']
        #         relevance = sample['relevance'].to(device)
        #
        #         # Forward pass without backprop
        #         cosine_sim, score = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0),
        #                                                   token_type_ids, Enc_Q, Enc_C, relevance)
        #         loss = loss_fn(pruning_score_prob.squeeze(), relevance)
        #         dev_epoch_loss += loss.item()
        #
        # # Calculate average losses
        # dev_epoch_loss = dev_epoch_loss / dev_samples
        # # train_epoch_loss = train_epoch_loss / train_samples
        # print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}, '
        #       f'Dev Loss: {dev_epoch_loss:.4f}, Time: {time.time() - epoch_start_time:.2f}s')
        # if dev_epoch_loss < best_dev_loss:
        #     best_dev_loss = dev_epoch_loss
        #     best_epoch = epoch
        #     print(f'Saving best model at epoch {epoch + 1} with dev loss {best_dev_loss:.4f}')
        #     save_model_path = save_model_path + f'{epoch}.pt'
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'best_dev_loss': best_dev_loss
        #     }, save_model_path)

        # print(f'Best dev loss {best_dev_loss:.4f} was achieved on epoch {best_epoch + 1}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Load tokenizer and model

    parser.add_argument('--gpu', type=int, default=1, help="GPU device ID. Use -1 for CPU training")
    parser.add_argument('--out_path', type = str, default="/home/hi9115/GCN-IR/Rank_candidate_project_Option1/"
                                                          "data/reduced_candid_neighbors/10-22-2024-models/model_",
                        help="Path to save the model")
    args = parser.parse_args()
    print(args)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    model = CERModel()
    # question_entities = utils.load_question_entities()
    # checkpoint = torch.load("/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/reduced_candid_neighbors/models_new/fine_tune_lambda05_maxlen512_loss_0.0213_epoch_2.pt")
    # model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)

    # Load dataset
    train_dataset = SubgraphPruningDataset(
        '/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/reduced_candid_neighbors/relevance_train.tsv',
        '/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/entities.json',
        tokenizer, device)
    dev_dataset = SubgraphPruningDataset(
        '/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/reduced_candid_neighbors/relevance_dev.tsv',
        '/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/entities.json',
        tokenizer, device)
    test_dataset = SubgraphPruningDataset(
        '/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/reduced_candid_neighbors/relevance_test.tsv',
        '/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/entities.json',
        tokenizer, device)

    # DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()  # Regression loss (Mean Squared Error)

    # Train the model
    train_model(model, train_loader, dev_loader, optimizer, loss_fn, args.out_path
                )
    # sate_dict = model.state_dict()
    # torch.save(model.state_dict(), "/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/reduced_candid_neighbors/models/fine_tuned.pt")
