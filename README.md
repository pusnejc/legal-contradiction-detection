# Legal Contradiction Detection

Contracts are built from short clauses that state obligations and definitions, and it's easy for two clauses in the same document, or between a document and a proposed edit, to conflict without anyone noticing. This project explores automatic **contradiction detection between pairs of legal clauses**: given a *premise* (a clause from a contract) and a *hypothesis* (a candidate statement), decide whether the hypothesis **contradicts** or **entails** the premise. Beyond classification, it also explores a **retrieval** angle — given a new clause, automatically surface existing clauses in a document set that it might contradict. This is closer to how such a tool would actually be used in practice.

The project progresses through four stages, each building on the previous one's findings:

1. **Data exploration and cleaning** — inspecting label balance, premise/hypothesis length distributions, and duplicates; dropping the `NotMentioned` label (all such rows have an empty premise, which would let a model learn a trivial shortcut instead of the actual language of contradiction) and deduplicating exact repeated pairs.
2. **Traditional ML baseline** — TF-IDF features (with `p_`/`h_` token prefixes to distinguish premise from hypothesis) feeding Logistic Regression, LinearSVC, and Random Forest, each tuned via cross-validated grid/random search.
3. **Transformer fine-tuning** — fine-tuning `bert-base-uncased` and `nlpaueb/legal-bert-base-uncased` for binary sequence-pair classification, comparing a domain-general model against a legal-domain-pretrained one.
4. **Sentence-embedding retrieval** — fine-tuning `all-MiniLM-L6-v2` with `ContrastiveLoss` so that contradicting clause pairs land closer in embedding space, then using it for top-k semantic search over a clause corpus, with the Stage 3 LegalBERT classifier applied as a post-filter/re-ranker over the retrieved candidates.

All stages treat **Contradiction as the positive class** and report F1/precision/recall on it, since the dataset is imbalanced (Entailment is the majority class) and accuracy alone would be a weak measure of model usefulness.

## Results

Numbers below are the final held-out **test-set** metrics reported by the notebook (final model per stage, retrained on train+validation and evaluated once on test).

| Model | Test F1 (Contradiction) | Precision | Recall | Accuracy |
|---|---|---|---|---|
| Logistic Regression (TF-IDF, C=10) | 0.822 | 0.763 | 0.891 | 0.928 |
| LegalBERT (lr=1e-5, max_len=256) | **0.860** | 0.870 | 0.850 | 0.949 |
| all-MiniLM-L6-v2 (cosine similarity, lr=2e-5) | 0.831 | 0.848 | 0.814 | 0.939 |

- A majority-class baseline (always predicting Entailment) scores F1 = 0 on Contradiction, at ~0.81 accuracy — reflecting the class imbalance rather than any real skill.
- LegalBERT (domain-pretrained on legal text) outperformed a general-domain `bert-base-uncased` on validation (F1 0.927 vs 0.884) and was selected as the final Stage 3 model.
- **Retrieval (Stage 4), embeddings only, on the test set** (220 contradiction queries, corpus = test-set premises): Recall@1 = 0.009, Recall@5 = 0.045, Recall@10 = 0.086, Recall@20 = 0.173, Recall@50 = 0.423, MRR@50 = 0.039, median best rank = 25. Recall improves steadily with k but is low for small k, indicating embeddings alone are better suited to candidate generation (retrieving a shortlist) than to pinpointing the exact contradicting clause. This is the motivation for combining retrieval with the LegalBERT post-filter in a practical pipeline.

Dataset (after removing `NotMentioned` and deduplicating): 4,292 training pairs / 859 validation pairs (stratified 80/20 split of the provided training data) / 1,188 test pairs, with Contradiction making up about 20% of examples in each split.

## Repository contents

- [`contradiction_detection.ipynb`](contradiction_detection.ipynb) — the full pipeline: data exploration, TF-IDF baselines, transformer fine-tuning, and the embedding retrieval tool with evaluation.

**Note on data:** `train.jsonl` / `test.jsonl` (premise/hypothesis/label triples) are not included in this repository (see `.gitignore` and [Attribution](#attribution) below). To run the notebook, place your own `train.jsonl` and `test.jsonl` (same schema: `premise`, `hypothesis`, `label` ∈ {`Contradiction`, `Entailment`, `NotMentioned`}) in the repository root.

## Running the notebook

Requires Python 3.9+ and a CUDA GPU is recommended for Stages 3–4 (fine-tuning falls back to CPU automatically but will be slow).

```bash
pip install pandas numpy scikit-learn matplotlib seaborn \
            transformers datasets accelerate torch sentence-transformers
```

Then, with `train.jsonl` and `test.jsonl` in place, open and run `contradiction_detection.ipynb` top to bottom (e.g. `jupyter lab` / `jupyter notebook`, or via VS Code's notebook interface). Cells are ordered to run sequentially: later stages (transformer fine-tuning, embedding training) depend on dataframes and splits created in earlier cells.

## Attribution

This project started as a university course assignment. The dataset was provided by the course and is not redistributed here; all data analysis, modeling, and code in the notebook are my own work.
