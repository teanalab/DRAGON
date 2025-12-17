import csv
import json
from torch.utils.data import Dataset
import torch
import utils
from transformers import BertTokenizer, BertModel

class SubgraphPruningDataset(Dataset):
    def __init__(self, tsv_file, json_file, tokenizer, device, max_len=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.device = device
        self.bert.to(device)
        # Load JSON file containing linked entities
        with open(json_file, 'r') as json_f:
            self.query_entities = json.load(json_f)

        # Load TSV file containing question_id, candidate, and relevance
        with open(tsv_file, mode='r') as tsv_f:
            tsv_reader = csv.reader(tsv_f, delimiter='\t')
            next(tsv_reader)  # Skip the header
            for row in tsv_reader:
                question_id, candidate, relevance = row
                # Append each row (with question_id, candidate, relevance) into self.samples
                self.samples.append({
                    'question_id': question_id,
                    'candidate': candidate,
                    'relevance': float(relevance)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        question_id = sample['question_id']
        candidate = sample['candidate']
        relevance = sample['relevance']

        # Get the entities linked to the query from the JSON file
        query_entities = self.query_entities.get(question_id, [])
        # Apply utils.uri_tokens to each entity in the query_entities list
        query_entities_clean = [utils.uri_entities(entity) for entity in query_entities]
        # Flatten the list of clean entities into a single string
        query_entities_str = " ".join([" ".join(tokens) for tokens in query_entities_clean])

        # Apply utils.uri_tokens to the candidate entity as well
        candidate_clean = utils.uri_entities(candidate)
        candidate_str = " ".join(candidate_clean)  # Convert the clean candidates into a string

        # Create the input text by concatenating query entities with the candidate entity
        input_text = query_entities_str + " [SEP] " + candidate_str

        # Tokenize the input
        inputs = self.tokenizer(
            input_text,
            add_special_tokens=True,  # Add [CLS] and [SEP] tokens
            max_length=self.max_len,  # Truncate/pad to max length
            padding='max_length',
            truncation=True,
            return_tensors='pt'  # Return PyTorch tensors
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        token_type_ids = inputs['token_type_ids'].squeeze()

        input_Q = self.tokenizer(
            query_entities_str,
            add_special_tokens=True,  # Add [CLS] and [SEP] tokens
            max_length=self.max_len,  # Truncate/pad to max length
            padding='max_length',
            truncation=True,
            return_tensors='pt'  # Return PyTorch tensors
        )
        input_ids_Q = input_Q['input_ids'].to(self.device)
        attention_mask_Q = input_Q['attention_mask'].to(self.device)
        token_type_ids_Q = input_Q['token_type_ids'].squeeze().to(self.device)
        Enc_Q = self.bert(input_ids=input_ids_Q, attention_mask=attention_mask_Q, token_type_ids=token_type_ids_Q)

        input_C = self.tokenizer(
            candidate_str,
            add_special_tokens=True,  # Add [CLS] and [SEP] tokens
            max_length=self.max_len,  # Truncate/pad to max length
            padding='max_length',
            truncation=True,
            return_tensors='pt'  # Return PyTorch tensors
        )
        input_ids_C = input_C['input_ids'].to(self.device)
        attention_mask_C = input_C['attention_mask'].to(self.device)
        token_type_ids_C = input_C['token_type_ids'].squeeze().to(self.device)
        Enc_C = self.bert(input_ids=input_ids_C, attention_mask=attention_mask_C, token_type_ids=token_type_ids_C)


        # Convert relevance to a float tensor
        relevance = torch.tensor(relevance, dtype=torch.float)
        # question_id = torch.tensor(int(question_id))

        return {
            'question_id': question_id,
            'candidate': candidate,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'relevance': relevance,
            'Enc_Q': Enc_Q,
            'Enc_C': Enc_C
        }
