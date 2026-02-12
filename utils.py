import json
import pickle
import re
import torch
from collections import defaultdict
import torch.nn.functional as F

from gensim.models import Word2Vec, KeyedVectors
# import lukovnikov
from unidecode import unidecode
import numpy as np
import nltk
from transformers import BertTokenizer
kewer_path = '/home/hi9115/GCN-IR/Rank_candidate_project_Option1/GCN-model/GCN-model/data/kewer.bin'
qblink_splits_path = 'data/qblink/QBLink-{}.json'
# qblink_splits_path = 'data/qblink/QBlink-Filtered-{}.json'
# qblink_splits_path = 'data/QBLink-Upload/filtered/QBlink-{}.json'
# qblink_splits_path = 'results/1-29-2024/new_dataset.json'
# qblink_splits_path = 'results/1-29-2024/dataset_1121.json'
word_probs_path = '/home/hi9115/GCN-IR/Rank_candidate_project_Option1/GCN-model/GCN-model/data/word_probs'
enwiki_links_path = 'data/enwiki-links.json'
dbpedia_redirects_path = 'data/dbpedia-2021-09-kewer/redirects_lang=en_transitive.ttl'
tagme_entities_path = 'data/tagme-entities.json'
lukovnikov_entities_path = '/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/entities.json'
question_neighbors_path = 'data/dbpedia-neighbors-2021-09-tagme.json'
reduced_question_neighbors_path = '/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/reduced_candid_neighbors/dbpedia-neighbors-2021-09-tagme-new.json'
prune_candid_path = '/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/reduced_candid_neighbors/epoch-2-k100-maxlen512/dbpedia-neighbors-2021-09-tagme-new.json'
neighbor_triples1000_path = '/home/hi9115/GCN-IR/Rank_candidate_project_Option1/data/reduced_candid_neighbors/dbpedia-triples-new.json'
neighbor_triples_path = 'data/dbpedia-triples.json'
neighbor_triples_path_llama = '/data/hi9115/GCN-IR-backup/GCN-IR/data/dbpedia-triples.json'
neighbor_features_path = 'data/features/dbpedia-features-{}.json'
overlap_features_path = 'data/dbpedia-overlap-features-{}.json'
overlap_features_path_new = 'data/new-triples/dbpedia-overlap-features-{}.json'

kvmem_triples_path = 'data/kvmem-triples/{}.json'
question_embeddings_path = 'data/question-embeddings/embedded_{}.pkl'

a = 0.0003  # word weighting parameter a
rho_na = 0.15  # threshold for annotations
regexp_tokenizer = nltk.RegexpTokenizer(r"['\w]+")
use_lukovnikov = True


# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
pred_blacklist = set(pred.lower() for pred in [
    '<http://www.w3.org/2000/01/rdf-schema#seeAlso>',
    '<http://dbpedia.org/property/seeAlso>',
    '<http://www.w3.org/2002/07/owl#differentFrom>',
    '<http://dbpedia.org/property/urlname>',
    '<http://dbpedia.org/property/isbn>',
    '<http://dbpedia.org/property/issn>',
    '<http://dbpedia.org/ontology/isbn>',
    '<http://dbpedia.org/ontology/issn>',
    '<http://dbpedia.org/property/caption>',
    '<http://dbpedia.org/property/imageCaption>',
    '<http://dbpedia.org/property/photoCaption>',
    '<http://dbpedia.org/property/mapCaption>',
    '<http://dbpedia.org/property/staticImageCaption>',
    '<http://dbpedia.org/property/floatCaption>',
    '<http://dbpedia.org/property/group>',
    '<http://dbpedia.org/property/groupstyle>',
    '<http://dbpedia.org/property/style>',
    '<http://dbpedia.org/property/align>',
    '<http://dbpedia.org/property/width>',
    '<http://dbpedia.org/property/bgcolor>',
    '<http://dbpedia.org/property/direction>',
    '<http://dbpedia.org/property/headerAlign>',
    '<http://dbpedia.org/property/footerAlign>',
    '<http://dbpedia.org/property/headerBackground>',
    '<http://dbpedia.org/property/imagenamel>',
    '<http://dbpedia.org/property/imagenamer>',
    '<http://dbpedia.org/property/imageAlt>',
    '<http://dbpedia.org/property/voy>',
    '<http://dbpedia.org/property/wikt>',
    '<http://dbpedia.org/property/commons>',
    '<http://dbpedia.org/property/id>',
    '<http://dbpedia.org/property/text>',
    '<http://dbpedia.org/property/reason>',
    '<http://dbpedia.org/property/hideroot>',
    '<http://dbpedia.org/property/notes>',
    '<http://dbpedia.org/property/crewPhotoAlt>',
    '<http://dbpedia.org/property/signatureAlt>',
    '<http://dbpedia.org/property/title>',
    '<http://dbpedia.org/property/alt>',
    # 'name' predicates that shouldn't be used for names
    '<http://dbpedia.org/property/nativeName>',
    '<http://dbpedia.org/ontology/formerName>',
    '<http://dbpedia.org/property/names',
    # name predicates that can be used for names
    '<http://dbpedia.org/property/name>',
    '<http://dbpedia.org/property/fullname>',
    '<http://dbpedia.org/property/birthName>',
    '<http://dbpedia.org/property/longName>',
    '<http://dbpedia.org/property/conventionalLongName>',
    '<http://dbpedia.org/property/commonName>',
    '<http://dbpedia.org/property/altname>',
    '<http://dbpedia.org/property/glottorefname>',
    '<http://dbpedia.org/property/sname>',
    '<http://dbpedia.org/property/clubname>',
    '<http://dbpedia.org/property/alternativeNames>',
    '<http://dbpedia.org/property/officialName>',
    '<http://xmlns.com/foaf/0.1/name>',
    '<http://dbpedia.org/ontology/birthName>',
    '<http://dbpedia.org/ontology/longName>',
    '<http://xmlns.com/foaf/0.1/givenName>',
    '<http://xmlns.com/foaf/0.1/surname>',  # didn't find this one in data
    '<http://dbpedia.org/property/showName>',
    '<http://dbpedia.org/property/shipName>',
    '<http://dbpedia.org/property/unitName>',
    '<http://dbpedia.org/property/otherName>',
    '<http://dbpedia.org/property/otherNames>'
])


def add_bool_arg(parser, name, default=False):
    """Add a boolean command-line argument.

    Allows to use --name and --no-name to specify value for the argument "name".
    Taken from https://stackoverflow.com/a/31347222."""
    name_internal = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name_internal, action='store_true')
    group.add_argument('--no-' + name, dest=name_internal, action='store_false')
    parser.set_defaults(**{name_internal: default})


def load_kewer():
    """Load KEWER embeddings.

    :return: loaded embeddings
    """
    return Word2Vec.load(kewer_path)


def extract_entityv(kewer):
    """Extract entity embeddings only from KEWER embeddings.
    :type kewer: gensim.models.word2vec.Word2Vec
    """
    entityv = KeyedVectors(kewer.vector_size)
    entities = []
    weights = []
    for entity in kewer.wv.vocab:
        if entity.startswith("Q") or entity.startswith("<") and "Category:" not in entity:
            entities.append(entity)
            weights.append(kewer.wv[entity])
    entityv.add(entities, weights)
    entityv.init_sims()
    return entityv


def load_word_probs():
    word_probs = {}

    with open(word_probs_path) as f:
        for line in f:
            word, prob = line.rstrip('\n').split('\t')
            word_probs[word] = float(prob)

    return word_probs


def load_enwiki_links():
    with open(enwiki_links_path) as elf:
        return json.load(elf)


def load_qblink_split(split_name):
    with open(qblink_splits_path.format(split_name)) as qbsf:
        return json.load(qbsf)


def tokenize(text):
    return [str(token).lower() for token in nltk.word_tokenize(text, preserve_line=True)]


def embed_tokens(tokens, wordv, word_probs):
    embeddings = []
    for token in tokens:
        if token in wordv:
            if token in word_probs:
                weight = a / (a + word_probs[token])
            else:
                weight = 1.0
            embeddings.append(wordv[token] * weight)
        # else:
        #     print('token {} not in vocab'.format(token))
    assert embeddings
    return embeddings

'''
def embed_question_bert(question_text, embeddings, word_probs, tokenizer, bert):
    input_ids = tokenizer(question_text)['input_ids']
    question_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    a = 0.0003  # word weighting parameter a
    token_embeddings = embed_tokens(question_tokens, embeddings, word_probs)
    
    kewer_question_embedding = np.zeros(embeddings.vector_size, dtype=np.float32)
    
    for token_embedding in token_embeddings:
        kewer_question_embedding += token_embedding

    kewer_question_embedding = kewer_question_embedding / np.linalg.norm(kewer_question_embedding)
    
    input_ids_tensor = torch.as_tensor(input_ids)
    bert_emd = bert(input_ids_tensor.unsqueeze(dim = 0))['last_hidden_state']
    
    bert_question_embedding = torch.sum(bert_emd.squeeze(), dim = 0)
    bert_question_embedding = bert_question_embedding.detach().numpy()
    
    bert_question_embedding = bert_question_embedding / np.linalg.norm(bert_question_embedding)
    
    return kewer_question_embedding, bert_question_embedding
'''

def embed_question_bert(question_text, embeddings, word_probs, tokenizer):
    input_ids = tokenizer(question_text)['input_ids']
    question_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # word weighting parameter a
    token_embeddings = embed_tokens(question_tokens, embeddings, word_probs)
    
    kewer_question_embedding = np.zeros(embeddings.vector_size, dtype=np.float32)
    
    for token_embedding in token_embeddings:
        kewer_question_embedding += token_embedding

    kewer_question_embedding = kewer_question_embedding / np.linalg.norm(kewer_question_embedding)

    return input_ids, kewer_question_embedding

def embed_question(question_text, embeddings, word_probs, question_entities=None, use_el=False):
    question_tokens = tokenize(question_text)
    token_embeddings = embed_tokens(question_tokens, embeddings, word_probs)
    question_embedding = np.zeros(embeddings.vector_size, dtype=np.float32)
    for token_embedding in token_embeddings:
        question_embedding += token_embedding
    if use_el:
        for entity, rho in question_entities:
            if entity in embeddings:
                question_embedding += embeddings[entity] * rho
    question_embedding = question_embedding / np.linalg.norm(question_embedding)
    return question_embedding


def tokens_embeddable(tokens, embeddings):
    """Check if there are any tokens that have embeddings"""
    for token in tokens:
        if token in embeddings:
            return True
    return False


def embed_literal(literal_tokens, embeddings, word_probs):
    token_embeddings = embed_tokens(literal_tokens, embeddings, word_probs)
    literal_embedding = np.zeros(embeddings.vector_size, dtype=np.float32)
    for token_embedding in token_embeddings:
        literal_embedding += token_embedding
    literal_embedding = literal_embedding / np.linalg.norm(literal_embedding)
    return literal_embedding


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def word_overlap_score(big_text_tokens, small_text_tokens, word_probs):
    overlap_score = 0.0
    for token in small_text_tokens:
        if token in big_text_tokens:
            if token in word_probs:
                weight = a / (a + word_probs[token])
            else:
                weight = 1.0
            overlap_score += weight
    return overlap_score

def word_overlap_score_gcn(big_text_tokens, small_text_tokens):
    overlap_score = 0
    for token in small_text_tokens:
        if token in big_text_tokens:
            overlap_score += 1
            # if token in word_probs:
            #     weight = a / (a + word_probs[token])
            # else:
            #     weight = 1.0
            # overlap_score += weight
    return overlap_score


def literal_tokens(text):
    text = (text.replace(r'\"', '"').replace(r'\t', '\t').replace(r'\b', '\b').
            replace(r'\n', '\n').replace(r'\r', '\r').replace(r'\f', '\f'))  # unescape characters

    if len(text) == 0:
        return []

    text = unidecode(text)  # remove accents
    text = text.replace('_', ' ')
    text = re.sub("([A-Z][a-z']+)", r' \1 ', text).strip()  # separate CamelCase

    tokens = regexp_tokenizer.tokenize(text)
    norm_tokens = []
    for token in tokens:
        token = re.sub(r"'s$", "", token)
        token = token.replace("'", "")
        norm_tokens.append(token.lower())
    return norm_tokens


def uri_tokens(uri):
    text = uri[uri.rfind('/') + 1:-1]
    text = text.replace("Category:", "", 1)
    return literal_tokens(text)

def uri_entities(uri):
    entity_name = uri[uri.rfind('/') + 1:-1]
    entity_name = entity_name.replace("Category:", "", 1)
    return entity_name

def dbpedia_redirects():
    redirects = {}
    with open(dbpedia_redirects_path) as f:
        for line in f:
            if not line.startswith('#'):
                subj, pred, obj = line.split(maxsplit=2)
                obj = obj[:obj.rfind('.')].strip()
                redirects[subj] = obj
    return redirects


def title2uri(title):
    return f"<http://dbpedia.org/resource/{title.replace(' ', '_')}>"


def load_question_entities(use_lukovnikov=True):
    if use_lukovnikov:
        with open(lukovnikov_entities_path) as entities_file:
            lukovnikov_entities = json.load(entities_file)
        question_entities = defaultdict(list)
        for question_id, entities in lukovnikov_entities.items():
            for entity in entities:
                    question_entities[question_id].append((entity, 1.0))
    else:
        with open(tagme_entities_path) as entities_file:
            tagme_entities = json.load(entities_file)

        question_entities = defaultdict(list)

        for question_id, entities in tagme_entities.items():
            for entity in entities:
                if entity['rho'] >= rho_na:
                    question_entities[question_id].append((title2uri(entity['title']), entity['rho']))

    return question_entities


def load_previous_answers():
    question_pervious_answers = defaultdict(list)

    for split in ['train', 'test', 'dev']:
        qblink_split = load_qblink_split(split)
        for sequence in qblink_split:
            if sequence['q1']['wiki_page']:
                q1_answer = f"<http://dbpedia.org/resource/{sequence['q1']['wiki_page']}>"
                question_pervious_answers[str(sequence['q2']['t_id'])].append(q1_answer)
                question_pervious_answers[str(sequence['q3']['t_id'])].append(q1_answer)
            if sequence['q2']['wiki_page']:
                q2_answer = f"<http://dbpedia.org/resource/{sequence['q2']['wiki_page']}>"
                question_pervious_answers[str(sequence['q3']['t_id'])].append(q2_answer)

    return question_pervious_answers


def load_question_neighbors():
    with open(question_neighbors_path) as qnf:
        return json.load(qnf)

def load_question_neighbors_new():
    with open(reduced_question_neighbors_path) as qnf:
        return json.load(qnf)

def load_prune_candid():
    with open(prune_candid_path) as qnf:
        return json.load(qnf)

def load_neighbor_triples_new():
    with open(neighbor_triples1000_path) as ntf:
        return json.load(ntf)

def load_neighbor_triples():
    with open(neighbor_triples_path) as ntf:
        return json.load(ntf)
def load_neighbor_triples_llama():
    with open(neighbor_triples_path_llama) as ntf:
        return json.load(ntf)

def load_neighbor_entities():
    question_neighbors = load_question_neighbors()
    neighbor_entities = set()
    for question, neighbors in question_neighbors.items():
        neighbor_entities.update(neighbors)
    return neighbor_entities


def load_split_features(split_name):
    with open(neighbor_features_path.format(split_name)) as nff:
        return json.load(nff)


def load_overlap_features(split_name):
    with open(overlap_features_path.format(split_name)) as off:
        return json.load(off)
def load_overlap_features_new(split_name):
    with open(overlap_features_path_new.format(split_name)) as off:
        return json.load(off)

def load_feature_inputs():
    with open('data/feature_inputs.pickle', 'rb') as handle:
        return pickle.load(handle)
    
'''  
def load_feature_inputs():
    with open('data/features/dbpedia-feature-inputs.pickle', 'rb') as handle:
        return pickle.load(handle)
'''  

def load_kvmem_triples(baseline_name):
    with open(kvmem_triples_path.format(baseline_name)) as ktf:
        return json.load(ktf)


def load_question_embeddings(split_name):
    with open(question_embeddings_path.format(split_name), 'rb') as qef:
        return pickle.load(qef)


def get_question_embedding(question_embeddings, question_id):
    return np.array(question_embeddings.loc[question_embeddings['ID'] == question_id, 'embedd'].iloc[0],
                    dtype=np.float32)


def div_pos(x, y):
    if y > 0:
        return x / y
    else:
        assert x == 0
        return x


def all_questions():
    questions = []
    for split in ['train', 'dev', 'test']:
        qblink_split = load_qblink_split(split)
        overlap_features = load_overlap_features(split)
        for sequence in qblink_split:
            for question in ['q1', 'q2', 'q3']:
                question_id = str(sequence[question]['t_id'])
                question_text = sequence[question]['quetsion_text']
                if question_id in overlap_features:
                    questions.append(question_text)
    return questions


# def init_bert_from_args(args):
#
#     from model_bert import ModelBert
#
#     return ModelBert(interaction=args.interaction, same_w=args.same_w, question_emb_dim=300,
#                      num_hidden=[int(h) for h in args.hidden.split(',')])
#
#
# def init_model_bert_from_args(args):
#
#     from model_bert_mult import ModelBertMult, ModelBertMultCAT
#
#     if args.concatenate == 'cat':
#         return ModelBertMultCAT(interaction=args.interaction, same_w=args.same_w, question_emb_dim=300,
#                          num_hidden=[int(h) for h in args.hidden.split(',')])
#     else:
#         return ModelBertMult(interaction=args.interaction, same_w=args.same_w, question_emb_dim=300,
#                          num_hidden=[int(h) for h in args.hidden.split(',')])

    
# def init_model_from_args(args):
#
#     if args.qemb == 'kewer':
#         from model_mult import ModelMult
#         return ModelMult(interaction=args.interaction, same_w=args.same_w, question_emb_dim=300,
#                          num_hidden=[int(h) for h in args.hidden.split(',')])
#
#     elif args.qemb == 'blstatic':
#         from model_mult import ModelMult
#         return ModelMult(interaction=args.interaction, same_w=args.same_w, question_emb_dim=4096,
#                          num_hidden=[int(h) for h in args.hidden.split(',')])
#     else:  # bldynamic
#         from model_bilstm_mult import ModelBiLSTMMult
#         return ModelBiLSTMMult(interaction=args.interaction, same_w=args.same_w,
#                                num_hidden=[int(h) for h in args.hidden.split(',')])
def jaccard_similarity(set1, set2):
    # Calculate the intersection and union of the two sets
    jacard_score = 0.0
    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set1, set):
        set2 = set(set2)

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    # Calculate the Jaccard coefficient
    if union == 0:
        return jacard_score  # To handle the case when both sets are empty
    jacard_score = intersection / union
    return jacard_score

def dot_product(s1, s2):
    cosine_similarity = torch.dot(s1, s2) / (torch.norm(s1) * torch.norm(s2))
    normalized_similarity = (cosine_similarity + 1) / 2
    return normalized_similarity.item()


# Example function to concatenate query and candidate entities and prepare input
def prepare_inputs(question_id, entities, candidates, max_len=128):
    query_entities = " ".join([uri_entities(e) for e in entities[question_id]])
    candidate_entities = candidates[question_id]

    inputs = []
    for candidate in candidate_entities:
        # Extract the entity name from the candidate URI
        candidate_name = uri_entities(candidate)
        # Concatenate the query entities and the candidate entity
        input_text = f"{query_entities} [SEP] {candidate_name}"

        # Tokenize the input
        encoded_input = tokenizer(
            input_text,
            add_special_tokens=True, # This will automatically add [CLS] at the beginning and [SEP] at the end
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs.append(encoded_input)
    return inputs

# Cosine similarity function
def cosine_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2, dim=-1)

# Function to compute pruning score
def compute_pruning_score(query_embedding, candidate_embedding, alpha, lambda_param=0.5):
    cos_sim = cosine_similarity(query_embedding, candidate_embedding)
    pruning_score = lambda_param * cos_sim + (1 - lambda_param) * alpha
    return pruning_score