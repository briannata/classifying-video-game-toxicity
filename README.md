
# Classifying Toxicity in Video Games

This project explores the use of transformer-based natural language processing (NLP) models to detect toxic language in online video game communities. Leveraging pre-trained models GPT-2 and BERT, the goal is to automate the classification of toxic versus non-toxic text from platforms such as Twitter, Twitch, and Discord.

## 🧠 Motivation

Toxicity in online gaming environments negatively affects players’ mental health and social experiences. While games offer fun and connection, toxic behavior—ranging from verbal abuse to spamming—is a growing issue. This project seeks to detect such toxicity using state-of-the-art language models, contributing to healthier gaming communities.

## 📚 Related Work

- Wessel Stoop (2021): Used a neural architecture combining GRUs and dense layers to detect patterns in toxic chat.
- University of Sydney (2021): Created the CONDA dataset from Dota 2 logs, enabling contextual toxicity classification.

## 📊 Dataset

The dataset includes labeled text samples from:
- **Twitter** (tweets with gaming keywords)
- **Twitch** (chat logs from game streams)
- **Discord** (discussions in large gaming communities)

Each entry is labeled as:
- `1`: Toxic
- `0`: Non-toxic

Labels include categories like severe toxicity, identity hate, and insults.

## 🧪 Methods

### Preprocessing

- Cleaned text: removed punctuation, numbers, excessive whitespace.
- Tokenization:
  - GPT-2: Byte Pair Encoding (BPE)
  - BERT: WordPiece tokenization
- Train-test split: 80/20, stratified
- Addressed class imbalance using ROC-AUC metrics.

### Models

Two transformer models were compared:
- **GPT-2** (small variant, 120M parameters)
- **BERT** (base model)

Both models were tested:
- Without fine-tuning
- With full fine-tuning
- With LoRA fine-tuning (BERT only)

## 📈 Results

| Model      | Fine-Tuning | Accuracy | Precision | Recall | AUROC  |
|------------|-------------|----------|-----------|--------|--------|
| GPT-2      | None        | 0.7923   | 0.0526    | 0.0088 | 0.4966 |
| GPT-2      | Full        | 0.9260   | 0.7614    | 0.9103 | 0.9199 |
| BERT       | None        | 0.8062   | 0.0000    | 0.0000 | 0.6519 |
| BERT       | Full        | 0.9089   | 0.7179    | 0.8458 | 0.9511 |
| BERT       | LoRA        | 0.9073   | 0.7036    | 0.8922 | 0.9616 |

### Insights

- Fine-tuning is essential for high performance.
- GPT-2 showed higher precision and recall post-fine-tuning.
- BERT with LoRA offered strong performance with fewer parameters—ideal for resource-constrained settings.

## ✅ Conclusion

This study confirms that fine-tuned transformer models can effectively detect toxicity in gaming discourse. GPT-2 offers better overall classification metrics, while BERT is more efficient and robust in terms of AUROC. Future directions include exploring other fine-tuning methods and datasets.

## 🛠 Technologies

- Python
- Hugging Face Transformers
- PyTorch
- Scikit-learn
- Pandas, NumPy

## 📂 References

- [GGWP Toxic Behavior Dataset](https://github.com/pl2599/GGWP-Toxic-Behavior)
- [BERT 101](https://huggingface.co/blog/bert-101)
- [Catching Cyberbullies](https://thegradient.pub/catching-cyberbullies-with-neural-networks/)
- [CONDA Dataset](https://aclanthology.org/2021.findings-acl.213/)
