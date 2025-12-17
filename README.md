# DRAGON: Conversational Entity Retrieval from Knowledge Graphs

**Accepted at WSDM 2026** | [Paper](https://github.com/teanalab/DRAGON)

**Authors:** Mona Zamiri, Alexander Kotov (Wayne State University)

Neural ranking architecture for conversational entity retrieval from knowledge graphs. DRAGON aggregates fine-grained relevance signals using Graph Convolutional Networks and multi-head attention.

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** PyTorch 2.4.1, DGL 2.4.0, Transformers 4.46.2

### Training Pipeline (Step-by-Step)

Execute the following scripts in order:

#### Step 1: Convert Triples to Dictionary Format
```bash
python convert_triple_to_dict.py \
  --input_tsv /path/to/triples.tsv \
  --output_json ./data/dict_train.json
```

#### Step 2: Train Sub-graph Pruning Model (Optional)
```bash
python Pruning_train.py \
  --gpu 0 \
  --out_path ./models/pruning_model
```

#### Step 3: Calculate Fine-grained Features
```bash
python Calculate_features.py \
  --gpu 0 \
  --output_file ./data/features_train.json
```

#### Step 4: Train DRAGON Ranking Model
```bash
python GCN_train.py \
  --gpu 0 \
  --score_file_dir /path/to/scores/ \
  --triple_file_dir /path/to/triples/ \
  --model_path ./models/dragon_model \
  --epochs 100
```

### Evaluation

```bash
python GCN_test.py \
  --gpu 0 \
  --model_path ./models/dragon_model.pt \
  --score_file_dir /path/to/scores/ \
  --triple_file_dir /path/to/triples/
```

---

## Data Format

### Input: Scores (JSON)
```json
[{"q_id": {"candidate": [s1, s2, ..., s12, n1_s1, ...], ...}}]
```

### Input: Triples (JSON)
```json
{"q_id": {"p": "positive_entity", "n": ["neg1", "neg2", ...]}}
```

### Output: Predictions (CSV)
```csv
question_id,candidate,score
q_001,entity_name,0.876
```

---

## Citation

```bibtex
@inproceedings{zamiri2026dragon,
  author = {Zamiri, Mona and Kotov, Alexander},
  title = {Conversational Entity Retrieval from a Knowledge Graph using Aggregation of Fine-grained Relevance Signals with Graph Convolutions and Self-Attention},
  booktitle = {WSDM 2026},
  year = {2026}
}
```

