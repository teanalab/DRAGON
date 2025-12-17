import argparse
import json
from tqdm import tqdm

import utils
import torch
from json_stream.writer import streamable_dict, streamable_list, StreamableList
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import numpy as np


# @lru_cache(maxsize=4098)
def calculate_features(qblink, bert_model, word_probs, kewer, tokenizer):
    pbar = tqdm(total=len(qblink))
    scores = {}
    for sequence in qblink:
        pbar.update(1)
        previous_answer_first_embedding, previous_answer_tokens_first = None, None
        previous_answer_second_embedding, previous_answer_tokens_second = None, None
        previous_answer_first = previous_answer_second = None
        for question in ['q1', 'q2', 'q3']:
            # if question not in sequence.keys():
            #     continue
            question_id = str(sequence[question]['t_id'])
            question_answer = f"<http://dbpedia.org/resource/{sequence[question]['wiki_page']}>"

            if (not sequence[question]['wiki_page'] or question_id not in question_candids
                    or question_answer not in question_candids[question_id]):
                continue

            question_neighbors_entities = question_candids[question_id]
            if question_answer in [entity for entity in question_neighbors_entities if entity not in neighbor_triples]:
                continue

            # Compute embeddings for the question
            question_text = sequence[question]['quetsion_text']
            question_tokens = set(utils.tokenize(question_text))
            question_embedding_kewer, question_embedding_bert = compute_question_embedding(question_text, bert_model, tokenizer, kewer, word_probs)
            question_embedding = torch.cat((question_embedding_bert, question_embedding_kewer), dim=0)  # Dimension: 768 + 300 = 1068

            # if question == 'q2' and 'q1' in sequence.keys():
            if question == 'q2':
                previous_answer_first = f"<http://dbpedia.org/resource/{sequence['q1']['wiki_page']}>"
                previous_answer_second = None
            # elif question == 'q3' and 'q1' in sequence.keys() and 'q2'in sequence.keys():
            elif question == 'q3':
                previous_answer_first = f"<http://dbpedia.org/resource/{sequence['q2']['wiki_page']}>"
                previous_answer_second = f"<http://dbpedia.org/resource/{sequence['q1']['wiki_page']}>"

            if previous_answer_first:
                previous_answer_first_embedding_kewer, previous_answer_first_embedding_bert = compute_entity_embedding_answer(previous_answer_first, bert_model, kewer)
                previous_answer_first_embedding = torch.cat((previous_answer_first_embedding_bert, previous_answer_first_embedding_kewer),
                                               dim=0)  # Dimension: 768 + 300 = 1068

                previous_answer_tokens_first = set(utils.uri_tokens(previous_answer_first))

            if previous_answer_second:

                previous_answer_second_embedding_kewer, previous_answer_second_embedding_bert = compute_entity_embedding_answer(previous_answer_second, bert_model, kewer)
                previous_answer_second_embedding = torch.cat((previous_answer_second_embedding_bert, previous_answer_second_embedding_kewer),
                                               dim=0)  # Dimension: 768 + 300 = 1068
                previous_answer_tokens_second = set(utils.uri_tokens(previous_answer_second))

            

            if question_id not in scores:
                scores[question_id] = {}

            for candidate_entity in question_neighbors_entities:
                if candidate_entity not in neighbor_triples or candidate_entity in scores[question_id]:
                    continue

                candidate_tokens = set(utils.uri_tokens(candidate_entity))
                candid_embedding_kewer, candid_embedding_bert = compute_entity_embedding_answer(candidate_entity, bert_model, kewer)
                candid_embedding = torch.cat((candid_embedding_bert, candid_embedding_kewer),
                                               dim=0)  # Dimension: 768 + 300 = 1068
                features = compute_features(question_tokens, candidate_tokens, question_embedding, candid_embedding,
                                            previous_answer_tokens_first, previous_answer_first_embedding,
                                            previous_answer_tokens_second, previous_answer_second_embedding)

                features = process_neighbor_features(candidate_entity, features, question_tokens,
                                                     previous_answer_tokens_first, previous_answer_first_embedding,
                                                     previous_answer_tokens_second, previous_answer_second_embedding,
                                                     question_embedding, bert_model, kewer)
                
                scores[question_id][candidate_entity] = features
            scores_list =convert_tensors_to_lists(scores)
            yield convert_tensors_to_lists(scores_list)
            scores = {}

    pbar.close()

def compute_entity_embedding_answer(entity, model, kewer):
    """Compute embeddings for a single entity using the model."""
    entity_name = utils.uri_entities(entity)
    embedding = model.encode(entity_name, convert_to_tensor=True, device=device)
    embedding = embedding / embedding.norm()

    
    kewer_question_embedding = np.zeros(kewer.wv.vector_size, dtype=np.float32)
    if entity in kewer.wv:
        kewer_question_embedding = kewer.wv[entity]

        kewer_question_embedding = kewer_question_embedding / np.linalg.norm(kewer_question_embedding)
    kewer_question_embedding = torch.tensor(kewer_question_embedding, dtype=torch.float32)
    kewer_question_embedding = kewer_question_embedding.to(device)

    return kewer_question_embedding, embedding

# @lru_cache(maxsize=4098)
# def compute_entity_embedding(entity, tknzr, bert_model, kewer):
def compute_entity_embedding(entity, model):
    """Compute embeddings for a single entity using the model."""
    entity_name = utils.uri_entities(entity)
    embedding = model.encode(entity_name, convert_to_tensor=True, device=device)
    embedding = embedding / embedding.norm()

    return embedding


# @lru_cache(maxsize=4098)
def compute_question_embedding(question_text, model, tokenizer, kewer, word_probs):
    """Compute the embeddings for the given question text."""
    
    embedding = model.encode(question_text, convert_to_tensor=True, device=device)
    embedding = embedding / embedding.norm()

    encoded_text = model.tokenizer(question_text, padding=True, truncation=True, return_tensors="pt")
    question_tokens = model.tokenizer.convert_ids_to_tokens(encoded_text['input_ids'][0])
    token_embeddings = utils.embed_tokens(question_tokens, kewer.wv, word_probs)

    kewer_question_embedding = np.zeros(kewer.wv.vector_size, dtype=np.float32)
    for token_embedding in token_embeddings:
        kewer_question_embedding += token_embedding

    # Normalize KEWER embedding
    kewer_question_embedding = kewer_question_embedding / np.linalg.norm(kewer_question_embedding)

    # Convert SBERT tensor to NumPy array
    kewer_question_embedding = torch.tensor(kewer_question_embedding, dtype=torch.float32)
    kewer_question_embedding = kewer_question_embedding.to(device)
    return kewer_question_embedding, embedding

# @lru_cache(maxsize=4098)



# @lru_cache(maxsize=4098)
def compute_features(question_tokens, candidate_tokens, question_embedding, candidate_embedding,
                     prev_answer_tokens_first, prev_answer_first_embedding, prev_answer_tokens_second,
                     prev_answer_second_embedding):
    """Compute features based on token overlaps and embeddings."""
    features = [0.0] * 12

    features[0] = utils.jaccard_similarity(question_tokens, candidate_tokens, word_probs)
    features[6] = utils.dot_product(question_embedding, candidate_embedding) if candidate_embedding is not None else 0.

    if prev_answer_tokens_first is not None:
        features[4] = utils.jaccard_similarity(prev_answer_tokens_first, candidate_tokens, word_probs)
        features[10] = utils.dot_product(prev_answer_first_embedding,
                                        candidate_embedding) if candidate_embedding is not None else 0.

    if prev_answer_tokens_second is not None:
        features[5] = utils.jaccard_similarity(prev_answer_tokens_second, candidate_tokens, word_probs)
        features[11] = utils.dot_product(prev_answer_second_embedding,
                                        candidate_embedding) if candidate_embedding is not None else 0.

    return features


# @lru_cache(maxsize=4098)

def process_neighbor_features(candidate_entity, features, question_tokens, prev_answer_tokens_first,
                              prev_answer_first_embedding, prev_answer_tokens_second, prev_answer_second_embedding,
                              question_embedding, bert_model, kewer):
    """Process features related to the neighbors of a candidate entity."""
    neighbor_types = ['lit', 'cat', 'subj', 'obj']

    for neighbor_type in neighbor_types:
        if neighbor_type in neighbor_triples[candidate_entity]:
            if neighbor_type in ['lit', 'subj', 'obj']:
                neighbors_list_list = neighbor_triples[candidate_entity][neighbor_type]
                lit_all = [item for subneighbor in neighbors_list_list for item in subneighbor]
                neighbors = set(lit_all)
            else:
                neighbors = neighbor_triples[candidate_entity][neighbor_type]

            for item in neighbors:
                # Process different types of neighbors distinctly
                if neighbor_type == 'lit':
                    feat_pred = [0.0] * 12
                    feat_lit = [0.0] * 12
                    if item.startswith('<http:'):
                        pred = item
                        pred_tokens = set(utils.uri_tokens(pred))
                        # Compute feature similarities
                        feat_pred[2] = utils.jaccard_similarity(pred_tokens, question_tokens, word_probs)
                        feat_pred[4] = utils.jaccard_similarity(pred_tokens,
                                                                prev_answer_tokens_first,
                                                                word_probs) if prev_answer_tokens_first else 0.0
                        feat_pred[5] = utils.jaccard_similarity(pred_tokens,
                                                                prev_answer_tokens_second,
                                                                word_probs) if prev_answer_tokens_second else 0.0

                        
                        pred_embedding_bert = compute_entity_embedding(pred, bert_model)
                        if pred in kewer.wv:
                            pred_embedding_kewer = kewer.wv[pred]
                            pred_embedding_kewer = pred_embedding_kewer / np.linalg.norm(pred_embedding_kewer)
                        else:
                            pred_embedding_kewer = np.zeros(kewer.wv.vector_size, dtype=np.float32)

                        pred_embedding_kewer = torch.tensor(pred_embedding_kewer, dtype=torch.float32)
                        pred_embedding_kewer = pred_embedding_kewer.to(device)

                        pred_embedding = torch.cat((pred_embedding_bert, pred_embedding_kewer),
                                               dim=0)  # Dimension: 768 + 300 = 1068
                        feat_pred[8] = utils.dot_product(question_embedding, pred_embedding)
                        feat_pred[10] = utils.dot_product(prev_answer_first_embedding,
                                                          pred_embedding) if prev_answer_first_embedding is not None else 0.0
                        feat_pred[11] = utils.dot_product(prev_answer_second_embedding,
                                                          pred_embedding) if prev_answer_second_embedding is not None else 0.0

                        features.extend(feat_pred)
                    else:
                        lit_tokens = set(item.split(' '))
                        feat_lit[1] = utils.jaccard_similarity(lit_tokens, question_tokens, word_probs)
                        feat_lit[4] = utils.jaccard_similarity(lit_tokens,
                                                      prev_answer_tokens_first, word_probs) if prev_answer_tokens_first else 0.0
                        feat_lit[5] = utils.jaccard_similarity(lit_tokens,
                                                      prev_answer_tokens_second, word_probs) if prev_answer_tokens_second else 0.0

                        lit_embedding_kewer, lit_embedding_bert = compute_literal_embedding(lit_tokens, bert_model, kewer)
                        lit_embedding = torch.cat((lit_embedding_bert, lit_embedding_kewer),
                                               dim=0)  # Dimension: 768 + 300 = 1068
                        feat_lit[7] = utils.dot_product(question_embedding, lit_embedding)
                        feat_lit[10] = utils.dot_product(prev_answer_first_embedding,
                                               lit_embedding) if prev_answer_first_embedding is not None else 0.0
                        feat_lit[11] = utils.dot_product(prev_answer_second_embedding,
                                               lit_embedding) if prev_answer_second_embedding is not None else 0.0

                        features.extend(feat_lit)


                    # # Additional processing for literal text (e.g., words in literal text)


                elif neighbor_type == 'cat':
                    feat_cat = [0.0] * 12

                    cat = item[1]
                    cat_tokens = set(utils.uri_tokens(cat))

                    feat_cat[3] = utils.jaccard_similarity(cat_tokens, question_tokens, word_probs)
                    feat_cat[4] = utils.jaccard_similarity(cat_tokens,
                                                  prev_answer_tokens_first, word_probs) if prev_answer_tokens_first else 0.0
                    feat_cat[5] = utils.jaccard_similarity(cat_tokens,
                                                  prev_answer_tokens_second, word_probs) if prev_answer_tokens_second else 0.0

                    # cat_embedding = compute_entity_embedding(cat, tknzr, bert_model, kewer)
                    cat_embedding_bert = compute_entity_embedding(cat, bert_model)
                    if cat in kewer.wv:
                        cat_embedding_kewer = kewer.wv[cat]
                        cat_embedding_kewer = cat_embedding_kewer / np.linalg.norm(cat_embedding_kewer)
                    else:
                        cat_embedding_kewer = np.zeros(kewer.wv.vector_size, dtype=np.float32)

                    cat_embedding_kewer = torch.tensor(cat_embedding_kewer, dtype=torch.float32)
                    cat_embedding_kewer = cat_embedding_kewer.to(device)

                    cat_embedding = torch.cat((cat_embedding_bert, cat_embedding_kewer),
                                              dim=0)  # Dimension: 768 + 300 = 1068
                    feat_cat[9] = utils.dot_product(question_embedding, cat_embedding)
                    feat_cat[10] = utils.dot_product(prev_answer_first_embedding,
                                           cat_embedding) if prev_answer_first_embedding is not None else 0.0
                    feat_cat[11] = utils.dot_product(prev_answer_second_embedding,
                                           cat_embedding) if prev_answer_second_embedding is not None else 0.0

                    features.extend(feat_cat)

                elif neighbor_type in ['subj', 'obj']:
                    feat_pred = [0.0] * 12
                    if not item.startswith('<http://dbpedia.org/resource/'):
                        pred = item
                        pred_tokens = set(utils.uri_tokens(pred))
                    
                        feat_pred[2] = utils.jaccard_similarity(pred_tokens, question_tokens, word_probs)
                        feat_pred[4] = utils.jaccard_similarity(pred_tokens,
                                                      prev_answer_tokens_first, word_probs) if prev_answer_tokens_first else 0.0
                        feat_pred[5] = utils.jaccard_similarity(pred_tokens,
                                                      prev_answer_tokens_second, word_probs) if prev_answer_tokens_second else 0.0

                        entity_embedding_bert = compute_entity_embedding(pred, bert_model)
                        if pred in kewer.wv:
                            entity_embedding_kewer = kewer.wv[pred]
                            entity_embedding_kewer = entity_embedding_kewer / np.linalg.norm(entity_embedding_kewer)
                        else:
                            entity_embedding_kewer = np.zeros(kewer.wv.vector_size, dtype=np.float32)

                        entity_embedding_kewer = torch.tensor(entity_embedding_kewer, dtype=torch.float32)
                        entity_embedding_kewer = entity_embedding_kewer.to(device)
                        entity_embedding = torch.cat((entity_embedding_bert, entity_embedding_kewer),
                                              dim=0)  # Dimension: 768 + 300 = 1068
                        feat_pred[8] = utils.dot_product(question_embedding, entity_embedding)
                        feat_pred[10] = utils.dot_product(prev_answer_first_embedding,
                                               entity_embedding) if prev_answer_first_embedding is not None else 0.0
                        feat_pred[11] = utils.dot_product(prev_answer_second_embedding,
                                               entity_embedding) if prev_answer_second_embedding is not None else 0.0

                        features.extend(feat_pred)
                    else:
                        feat_ent = [0.0] * 12
                        ent_tokens = set(utils.uri_tokens(item))

                        feat_ent[0] = utils.jaccard_similarity(ent_tokens, question_tokens, word_probs)
                        feat_ent[4] = utils.jaccard_similarity(ent_tokens,
                                                      prev_answer_tokens_first,
                                                      word_probs) if prev_answer_tokens_first else 0.0
                        feat_ent[5] = utils.jaccard_similarity(ent_tokens,
                                                      prev_answer_tokens_second,
                                                      word_probs) if prev_answer_tokens_second else 0.0

                        entity_embedding_bert = compute_entity_embedding(item, bert_model)
                        if item in kewer.wv:
                            entity_embedding_kewer = kewer.wv[item]
                            entity_embedding_kewer = entity_embedding_kewer / np.linalg.norm(entity_embedding_kewer)
                        else:
                            entity_embedding_kewer = np.zeros(kewer.wv.vector_size, dtype=np.float32)

                        entity_embedding_kewer = torch.tensor(entity_embedding_kewer, dtype=torch.float32)
                        entity_embedding_kewer = entity_embedding_kewer.to(device)
                        entity_embedding = torch.cat((entity_embedding_bert, entity_embedding_kewer),
                                              dim=0)  # Dimension: 768 + 300 = 1068
                        feat_ent[6] = utils.dot_product(question_embedding, entity_embedding)
                        feat_ent[10] = utils.dot_product(prev_answer_first_embedding,
                                               entity_embedding) if prev_answer_first_embedding is not None else 0.0
                        feat_ent[11] = utils.dot_product(prev_answer_second_embedding,
                                               entity_embedding) if prev_answer_second_embedding is not None else 0.0

                        features.extend(feat_ent)
    return features


def compute_literal_embedding(lit_tokens, model, kewer):
    """Compute the SBERT embedding for literal tokens."""
    # Join tokens into a single string for SBERT input
    lit_text = " ".join(lit_tokens)

    # Encode using SBERT
    lit_embedding = model.encode(lit_text, convert_to_tensor=True, device=device)
    # Normalize the embedding
    lit_embedding = lit_embedding / lit_embedding.norm()
    kewer_lit_embedding = np.zeros(kewer.wv.vector_size, dtype=np.float32)
    
    if utils.tokens_embeddable(lit_tokens, kewer.wv):
        kewer_lit_embedding = utils.embed_literal(lit_tokens, kewer.wv, word_probs)
        kewer_lit_embedding = kewer_lit_embedding / np.linalg.norm(kewer_lit_embedding)

    
    kewer_lit_embedding = torch.tensor(kewer_lit_embedding, dtype=torch.float32)
    kewer_lit_embedding = kewer_lit_embedding.to(device)


    return kewer_lit_embedding, lit_embedding



# @lru_cache(maxsize=4098)
def convert_tensors_to_lists(d) -> list|dict:
    """Recursively converts all tensors in a dictionary to lists."""
    if isinstance(d, dict):
        return {k: convert_tensors_to_lists(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_tensors_to_lists(i) for i in d]
    elif isinstance(d, torch.Tensor):
        d = d.type(torch.float16)
        return d.tolist()
    else:
        return d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='dev', choices=['train', 'dev', 'test'])
    parser.add_argument('--gpu', type=int, default=0, help="GPU device ID. Use -1 for CPU training")
    parser.add_argument('--outpath')
    args = parser.parse_args()
    print(args)

    if args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1').to(device)  # Choose a suitable SBERT model
    model.max_seq_length = 512  # Set max sequence length to 512
    model = model.to(device)
    qblink_split = utils.load_qblink_split(args.split)
    word_probs = utils.load_word_probs()
    kewer = utils.load_kewer()
    neighbor_triples = utils.load_neighbor_triples_new()
    question_candids = utils.load_prune_candid()

    word_probs = utils.load_word_probs()
    feature_list = calculate_features(qblink_split, model, word_probs, kewer, tokenizer)
    streamable_feature_list:StreamableList = streamable_list(feature_list)
    with open(f"{args.outpath}_{args.split}.json", 'w') as f:
        json.dump(streamable_feature_list, f, indent=4)

    print(f"Scores saved to {args.outpath}_{args.split}.json")
