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

# Step 18 - gradient_descent_step
def gradient_descent_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lr: float, l2_lambda: float) -> tuple:
    # TODO: Run one full-batch gradient descent update; return (w_new, b_new, loss).
    y_prob = logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, y_prob, w, l2_lambda)
    dw, db = logistic_gradients(X, y, y_prob, w, l2_lambda)

    w_new = w - lr * dw
    b_new = b - lr * db

    return (w_new, b_new, loss)

# Step 19 - train_logistic_regression
def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float, l2_lambda: float, n_epochs: int) -> tuple:
    # TODO: Initialize params and run n_epochs of full-batch GD, recording loss...
    n_features = X.shape[1]
    w, b = initialize_logistic_params(n_features)
    losses = []
    for _ in range(n_epochs):
        w, b, loss = gradient_descent_step(X, y, w, b, lr, l2_lambda)
        losses.append(loss)

    return (w, b, losses)

# Step 20 - predict_labels
def predict_labels(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert predicted probabilities into hard binary labels.

    Args:
        proba: 1-D array of probabilities in [0, 1], shape (N,).
        threshold: Decision threshold; proba >= threshold maps to 1.

    Returns:
        Integer array of shape (N,) with values in {0, 1}.
    """
    # TODO: Convert probabilities to hard binary labels via the threshold...
    HBL = []
    for i in range(len(proba)):
        HBL.append(int(proba[i] >= threshold))

    return np.array(HBL)

# Step 21 - confusion_counts
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    # TODO: Return the four confusion-matrix counts (tp, fp, tn, fn) as Python ints
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return (tp, fp, tn, fn)

# Step 22 - metrics_from_counts
def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    # TODO: Derive precision, recall, F1, and accuracy from confusion counts...
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    acc = 0.0
    if tp + fp != 0: precision = tp / (tp + fp)
    if tp + fn != 0: recall = tp / (tp + fn)
    if precision+recall != 0: f1 = (2 * (precision*recall)) / (precision+recall)
    if tp+fp+fn+tn != 0: acc = (tp+tn) / (tp+fp+fn+tn)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': acc
    }

# Step 23 - tune_decision_threshold
def tune_decision_threshold(y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray = None) -> tuple:
    # TODO: Find the decision threshold that maximizes F1 on validation data.
    if thresholds is None: thresholds = np.linspace(0.0, 1.0, 101)

    best_f1 = -1.0
    best_t = thresholds[0]

    for t in thresholds:
        y_pred = predict_labels(proba, threshold=float(t))
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)
        metrics = metrics_from_counts(tp, fp, tn, fn)

        if metrics['f1'] > best_f1:
            best_f1 = float(metrics['f1'])
            best_t = float(t)

    return (best_t, best_f1)

# Step 24 - evaluate_predictions
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # TODO: Bundle confusion counts and classification metrics into one report dict
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)
    return {
        'tp':tp,
        'fp':fp,
        'tn':tn,
        'fn':fn,
        'precision':metrics['precision'],
        'recall':metrics['recall'],
        'f1':metrics['f1'],
        'accuracy':metrics['accuracy']
    }

# Step 25 - vectorize_texts
def vectorize_texts(texts: list, vocab: dict, idf: np.ndarray) -> np.ndarray:
    # TODO: Clean, tokenize, BoW, and TF-IDF transform a list of raw strings.
    tokenized = tokenize_corpus(texts)
    bow = corpus_to_bow_matrix(tokenized, vocab)
    return transform_tfidf(bow, idf)

# Step 26 - predict_text
def predict_text(text: str, vocab: dict, idf: np.ndarray, w: np.ndarray, b: float, threshold: float = 0.5) -> int:
    """Label a single raw message with the fitted classifier.

    Args:
        text: Raw input string.
        vocab: Fitted word -> column index map.
        idf: Fitted IDF vector, shape (V,).
        w: Logistic weight vector, shape (V,).
        b: Logistic bias scalar.
        threshold: Decision threshold for the positive class.

    Returns:
        Predicted label as int 0 or 1.
    """
    # TODO: label a single unseen raw message using fitted model artifacts
    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b)
    labels = predict_labels(proba, threshold)
    return int(labels[0])

# Step 27 - collect_prediction_errors (not yet solved)
# TODO: implement

