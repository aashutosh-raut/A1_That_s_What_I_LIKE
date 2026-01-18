# A1: That’s What I LIKE – Semantic Search with Word Embeddings

**Course:** AT82.05 Artificial Intelligence: Natural Language Understanding (NLU)
**Assignment:** A1 – That’s What I LIKE

---

## Introduction

This assignment focuses on building a **semantic search system** that retrieves the most contextually similar paragraphs from a text corpus given a user query (e.g., *“Harry Potter”*).
Instead of relying on keyword matching, the system uses **word embeddings** to capture semantic meaning and compares texts based on contextual similarity.

The project builds upon code provided in class and includes implementations of:

* **Word2Vec (Skip-gram)**
* **Word2Vec (Skip-gram with Negative Sampling)**
* **GloVe (from scratch)**
* **Pre-trained GloVe (via Gensim)**

The models are trained on a real-world corpus and evaluated using:

* Word analogy tasks
* Correlation with human similarity judgments
* A semantic search application using paragraph embeddings

---

## Summary of the Word2Vec Paper

**Efficient Estimation of Word Representations in Vector Space**
*Mikolov et al., 2013*

The Word2Vec framework learns dense vector representations of words. The **Skip-gram** model trains embeddings by predicting surrounding context words given a target word within a fixed window.

A key contribution is **Negative Sampling**, which reduces computational cost by updating only a small number of negative examples instead of the full vocabulary. This makes training scalable while preserving semantic quality.

The resulting embeddings capture both **semantic and syntactic relationships**, enabling meaningful vector arithmetic such as word analogy tasks.

---

## Summary of the GloVe Paper

**GloVe: Global Vectors for Word Representation**
*Pennington et al., 2014*

GloVe is a **count-based** embedding method that leverages **global word co-occurrence statistics**. It constructs a word–word co-occurrence matrix and learns embeddings by minimizing a weighted least-squares loss over the logarithm of co-occurrence counts.

Unlike Word2Vec’s local prediction objective, GloVe explicitly models global statistics. The paper demonstrates that **ratios of co-occurrence probabilities** encode semantic relationships effectively.

In this assignment, GloVe is implemented from scratch and compared with a **pre-trained GloVe model** using Gensim.

---

## Project Overview

This repository contains all components required for Assignment 1, including model implementations, experiments, evaluations, and a semantic search application.

---

## Repository Structure

```
.
├── 01 - Word2Vec (Skipgram) copy.ipynb
├── 02 - Word2Vec (Neg Sampling).ipynb
├── 03 - GloVe from Scratch.ipynb
├── 04 - GloVe (Gensim).ipynb
├── app/
│   └── web application files
└── README.md
```

---

## Dataset Sources

### Training Corpus

* **NLTK Brown Corpus**
  Source: [https://www.nltk.org/](https://www.nltk.org/)
  A balanced collection of English text from multiple genres.

### Word Analogy Dataset

* **Mikolov et al. Word Analogy Dataset**
  [https://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt](https://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt)

### Word Similarity Dataset

* **WordSim-353**
  [http://alfonseca.org/eng/research/wordsim353.html](http://alfonseca.org/eng/research/wordsim353.html)

All datasets are publicly available and properly credited.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-github-repo-url>
cd A1_That_s_What_I_LIKE
```

### 2. Install Required Libraries

```bash
pip install numpy torch nltk gensim scipy pandas flask
```

### 3. Download NLTK Data

```python
import nltk
nltk.download('brown')
```

### 4. Run the Notebooks

Open each notebook in Jupyter and run all cells:

* `01 - Word2Vec (Skipgram) copy.ipynb`
* `02 - Word2Vec (Neg Sampling).ipynb`
* `03 - GloVe from Scratch.ipynb`
* `04 - GloVe (Gensim).ipynb`

---

## Experiments and Results

### Model Training Configuration

All custom models were trained using:

* **Corpus:** Brown corpus
* **Embedding size:** 8
* **Window size:** 2 (dynamically configurable)
* **Epochs:** 1000

The training pipeline supports dynamic modification of the context window size, with a default value of **2**, as required by the assignment.

---

### Word Analogy Evaluation

| Model           | Window Size | Training Loss | Training Time (s) | Syntactic Accuracy | Semantic Accuracy |
| --------------- | ----------: | ------------: | ----------------: | -----------------: | ----------------: |
| Skip-gram       |           2 |      7.503079 |        1494.55755 |           0.000000 |          0.000000 |
| Skip-gram (NEG) |           2 |       2.68448 |          4.104302 |           0.000000 |          0.000000 |
| GloVe (Scratch) |           2 |      3.249292 |        146.134839 |           0.000000 |          0.000000 |
| GloVe (Gensim)  |         100 |             – |                 – |           0.554487 |          0.894433 |

#### Discussion

The custom-trained models achieved **0% accuracy** on word analogy tasks, which is expected given the relatively small size and limited vocabulary of the Brown corpus.
The pre-trained GloVe model performs significantly better, highlighting the importance of **large-scale training data**.

---

### Word Similarity Evaluation (WordSim-353)

Spearman rank correlation was computed between model dot-product similarities and human similarity judgments.

| Model           | Spearman Correlation |
| --------------- | -------------------: |
| Skip-gram       |             0.040807 |
| Skip-gram (NEG) |             0.053529 |
| GloVe (Scratch) |             0.099987 |
| GloVe (Gensim)  |                    – |

#### Discussion

The low correlations for custom models reflect corpus size and domain limitations.
The stronger performance of the pre-trained GloVe model demonstrates the effectiveness of embeddings trained on **large and diverse datasets**.

---

## Semantic Search Application

A simple **web-based semantic search system** was implemented as part of the assignment:

* Query and paragraph embeddings are computed by **averaging word vectors**
* Similarity is measured using the **dot product**
* The system retrieves the **top 10 most similar contexts**

Due to limited training data and short embedding dimensionality, results may not always be semantically perfect. However, the application demonstrates a complete **end-to-end pipeline** from embedding learning to real-world usage.

---

## Conclusion

This assignment provided hands-on experience with implementing and evaluating word embedding models from scratch. By comparing **Word2Vec** and **GloVe** approaches and applying them to a semantic search task, the strengths and limitations of different embedding techniques became clear.

The experiments highlight how **corpus size, data diversity, and training methodology** directly affect embedding quality and downstream performance.

---
