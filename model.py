"""
NumPy Text Classifier from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - clean_text
def clean_text(text: str) -> str:
    # TODO: Lowercase text and replace non-alphabetic chars with spaces
    clean = ""
    for char in text:
        if not char.isalpha():
            char = ' '
            clean += char
        else:
            char = char.lower()
            clean += char

    return clean.rstrip()

# Step 2 - tokenize
def tokenize(text: str) -> list:
    # TODO: Split cleaned text on whitespace into non-empty word tokens
    return [string for string in text.split() if len(string) != 0]

# Step 3 - tokenize_corpus
def tokenize_corpus(texts: list) -> list:
    # TODO: Apply clean_text and tokenize to every document so the full corpus becomes a list of token lists.
    token_list = []
    for text in texts:
        cleaned = clean_text(text)
        tokenized = tokenize(cleaned)
        token_list.append(tokenized)

    return token_list

# Step 4 - split_train_val_test_indices
import numpy as np
def split_train_val_test_indices(n_samples: int, val_fraction: float, test_fraction: float, seed: int = 0) -> tuple:
    # TODO: Produce shuffled index arrays that partition n_samples into train/val/test
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_samples)

    start = 0
    n_val = int(n_samples * val_fraction)
    end = n_val
    val_idx = shuffled[start:end]

    start = end
    n_test = int(n_samples * test_fraction)
    end += n_test
    test_idx = shuffled[start:end]

    start = end
    train_idx = shuffled[start:]

    return (train_idx, val_idx, test_idx)

# Step 5 - count_word_frequencies
def count_word_frequencies(tokenized_docs: list) -> dict:
    # TODO: Return a dict mapping each unique token to its total count...
    keys = []
    for doc in tokenized_docs:
        for token in doc:
            keys.append(token)

    key_map = {}
    set_of_keys = set(keys)
    for key in set_of_keys:
        key_map[key] = keys.count(key)

    return dict(sorted(key_map.items(), key=lambda item: item[1], reverse=True))

# Step 6 - build_vocabulary
def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    # TODO: Keep the top max_size most frequent words; map each to an index in [0, V).
    sorted_dict = sorted(word_counts, key=lambda w: (-word_counts[w], w))

    vocab = {}
    for v, word in enumerate(sorted_dict[:max_size]):
        vocab[word] = v

    return vocab

# Step 7 - tokens_to_bow
def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    # TODO: Convert one document's token list into a bag-of-words count vector...
    arr = []
    for key in vocab:
        arr.append((tokens.count(key)))

    return np.array(arr, dtype=float)

# Step 8 - corpus_to_bow_matrix
def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    # TODO: Stack per-document BoW vectors into a 2-D count matrix for a whole corpus.
    if not tokenized_docs:
        return np.ndarray(shape=(0, len(vocab)))

    matrix = []
    for tokens in tokenized_docs:
        row = []
        for key in vocab:
            row.append((tokens.count(key)))
        matrix.append(row)

    return np.array(matrix, dtype=float)

# Step 9 - compute_document_frequencies
def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    # TODO: Count docs where each term appears at least once (df, shape (V,))
    df = []
    N = bow_matrix.shape[0]
    V = bow_matrix.shape[1]
    for col in range(V):
        count = 0
        for row in range(N):
            if bow_matrix[row][col] > 0:
                count += 1
        df.append(count)
    return np.array(df)

# Step 10 - compute_idf
def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    # TODO: Compute smoothed IDF idf_j = log((n_docs + 1) / (df_j + 1)) + 1
    IDF = []
    for j in range(len(df)):
        IDF.append(np.log((n_docs + 1) / (df[j] + 1)) + 1)

    return np.array(IDF)

# Step 11 - transform_tfidf
def transform_tfidf(bow_matrix: np.ndarray, idf: np.ndarray) -> np.ndarray:
    # TODO: Multiply BoW counts by the fitted IDF vector to produce TF-IDF features.
    return bow_matrix * idf

# Step 12 - fit_tfidf
def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    # TODO: Fit IDF on the training BoW matrix by chaining DF and IDF.
    df = compute_document_frequencies(bow_train)
    n_docs = bow_train.shape[0]
    return compute_idf(df, n_docs)

# Step 13 - sigmoid
def sigmoid(z: np.ndarray) -> np.ndarray:
    # TODO: Map logits to probabilities with a numerically stable logistic sigmoid.
    log_prob = []
    for x in range(len(z)):
        log_prob.append(1 / (1 + np.exp(-z[x])))

    return np.array(log_prob)

# Step 14 - logistic_predict_proba
def logistic_predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    # TODO: Return P(y=1|x) for each row via linear scores and sigmoid
    z = X @ w + b
    return sigmoid(z)

# Step 15 - binary_cross_entropy
def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> float:
    # TODO: Compute mean binary cross-entropy plus L2 penalty on the weights.
    BCE = -np.mean([y_true[i] * np.log(y_prob[i]) + (1 - y_true[i]) * np.log(1 - y_prob[i]) for i in range(len(y_true))])
    return float(BCE + (l2_lambda * np.sum(w ** 2) / 2))

# Step 16 - logistic_gradients
def logistic_gradients(X: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> tuple:
    """Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.

    Args:
        X: Feature matrix of shape (N, D).
        y_true: Binary labels of shape (N,).
        y_prob: Predicted probabilities of shape (N,).
        w: Weight vector of shape (D,).
        l2_lambda: L2 regularization strength.

    Returns:
        Tuple (dw, db) with dw shape (D,) and db a float.
    """
    # TODO: Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.
    r = y_prob - y_true
    dw = ((X.T @ r) / X.shape[0]) + (l2_lambda * w)
    db = np.mean(r)

    return (dw, db)

# Step 17 - initialize_logistic_params
def initialize_logistic_params(n_features: int) -> tuple:
    # TODO: Return a zero weight vector of shape (n_features,) and bias 0.0
    return (np.zeros(shape=n_features), 0.0)

# Step 18 - gradient_descent_step (not yet solved)
# TODO: implement

# Step 19 - train_logistic_regression (not yet solved)
# TODO: implement

# Step 20 - predict_labels (not yet solved)
# TODO: implement

# Step 21 - confusion_counts (not yet solved)
# TODO: implement

# Step 22 - metrics_from_counts (not yet solved)
# TODO: implement

# Step 23 - tune_decision_threshold (not yet solved)
# TODO: implement

# Step 24 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 25 - vectorize_texts (not yet solved)
# TODO: implement

# Step 26 - predict_text (not yet solved)
# TODO: implement

# Step 27 - collect_prediction_errors (not yet solved)
# TODO: implement

