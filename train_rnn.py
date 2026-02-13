"""
train_rnn.py  
================================================================
Downloads the Bitext Customer Support dataset, applies proper
preprocessing and training proceduress,
trains a PyTorch bidirectional LSTM intent-classifier, evaluates
it with a classification report + confusion matrix, and saves all
artifacts to the  model/  directory.

Usage:
    python train_rnn.py
"""

import json, os, pickle, re, string, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")                       
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from collections import Counter

# ─── Attempt to import nltk for stopword removal ─────────────
# Keep interrogative words — they carry critical intent information
# (e.g. "where is my order" vs "place an order" both reduce to "order" without them)
KEEP_WORDS = {"where", "what", "how", "when", "which", "why", "can", "could",
              "would", "should", "want", "need", "track", "check", "cancel",
              "change", "get", "find", "status"}

try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import nltk
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english")) - KEEP_WORDS
    STEMMER = PorterStemmer()
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    STOP_WORDS = set()
    STEMMER = None

# ========================== COnfigurations ==============================================-=-=-=-=-=
SAVE_DIR       = "model"
RUNS_DIR       = os.path.join(SAVE_DIR, "runs")
VOCAB_SIZE     = 8000
MAX_SEQ_LEN    = 40
EMBED_DIM      = 128          # increased from 64 
HIDDEN_DIM     = 128          # (scaled for our task)
NUM_LSTM_LAYERS = 2           # deepe1 LSTM
DROPOUT        = 0.5          # as mam said in lectures
EPOCHS         = 100          # more epochs w/ early stopping 
BATCH_SIZE     = 64
LEARNING_RATE  = 1e-3
TEST_SPLIT     = 0.20         # 80/20 split 
PATIENCE       = 5            # for early stopping 
# ==============================================================================


# ─── TEXT PREPROCESSING (L 7) ──────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Apply text preprocessing steps as outlined in    L 7:
    1. Lowercasing
    2. Remove non-alphabetic characters
    3. Tokenisation
    4. Stopword removal
    5. Stemming
    """
    # 1. Lowercase
    text = text.lower().strip()

    # 2. Remove punctuation & non-alphabetic chars (   L 7 step 6)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 3-5. Tokenise → remove stopwords → stem
    tokens = text.split()
    if HAS_NLTK:
        tokens = [STEMMER.stem(w) for w in tokens if w not in STOP_WORDS]
    else:
        # basic stopword list fallback
        basic_stops = {"the", "a", "an", "is", "are", "was", "were", "i", "my",
                       "me", "we", "our", "you", "your", "it", "its", "to", "of",
                       "in", "on", "for", "with", "and", "or", "but", "not", "do",
                       "does", "did", "have", "has", "had", "be", "been", "am"}
        tokens = [w for w in tokens if w not in basic_stops]

    return " ".join(tokens)


# ─── VOCAB & ENCODING (   L 7    s 18-19) ────────────────────────────────────
def build_vocab(texts: list[str], vocab_size: int) -> dict[str, int]:
    """Build word -> index mapping.  0 = PAD, 1 = UNK."""
    counter: Counter = Counter()
    for t in texts:
        counter.update(t.split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(texts: list[str], vocab: dict[str, int], max_len: int) -> np.ndarray:
    """Tokenise + pad/truncate to fixed length (   L 7: pad_sequences)."""
    unk = vocab["<UNK>"]
    encoded = []
    for t in texts:
        ids = [vocab.get(w, unk) for w in t.split()][:max_len]
        ids += [0] * (max_len - len(ids))           # right-pad with 0
        encoded.append(ids)
    return np.array(encoded, dtype=np.int64)


# ─── LSTM MODEL (   L 7    s 20, 23 +    L 6     14) ────────────────────────
class IntentLSTM(nn.Module):
    """
    Embedding -> Bidirectional LSTM (stacked) -> Dropout -> Dense -> Softmax.
    Architecture follows    L 7 pattern (Embedding + LSTM + Dense)
    with improvements from    L 6 (Dropout regularisation).
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_classes: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)       #    L 6     21,    L 7     4
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        emb = self.embedding(x)                          # (B, T, E)
        _, (h, _) = self.lstm(emb)                       # h: (2*L, B, H)
        # take last layer forward + backward
        h = torch.cat((h[-2], h[-1]), dim=1)             # (B, 2H)
        return self.fc(self.dropout(h))


# ─── EARLY STOPPING (   L 6     10) ──────────────────────────────────────────
class EarlyStopping:
    """
    Stops training when val_loss does not improve for `patience` epochs.
    Restores best model weights (   L 6: restore_best_weights=True).
    """
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


# ─── SESSION VERSIONING ──────────────────────────────────────────────────────
def archive_session(history: dict, config: dict, final_val_acc: float,
                    final_val_loss: float, epochs_run: int,
                    classification_rpt: str):
    """
    Save a snapshot of this training run to model/runs/<timestamp>/
    and append a summary to model/runs/sessions_log.json for comparison.
    """
    os.makedirs(RUNS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(RUNS_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Copy all current model artifacts into the run folder
    for fname in os.listdir(SAVE_DIR):
        src = os.path.join(SAVE_DIR, fname)
        if os.path.isfile(src):
            shutil.copy2(src, run_dir)

    # Save the classification report as text
    with open(os.path.join(run_dir, "classification_report.txt"), "w") as f:
        f.write(classification_rpt)

    # ── Build session summary ────────────────────────────────────────────────
    session_info = {
        "run_id":         timestamp,
        "timestamp":      datetime.now().isoformat(),
        "epochs_run":     epochs_run,
        "max_epochs":     config.get("max_epochs", EPOCHS),
        "early_stopped":  epochs_run < config.get("max_epochs", EPOCHS),
        "final_train_acc": round(history["train_acc"][-1], 4),
        "final_val_acc":   round(final_val_acc, 4),
        "final_train_loss":round(history["train_loss"][-1], 4),
        "final_val_loss":  round(final_val_loss, 4),
        "best_val_loss":   round(min(history["val_loss"]), 4),
        "best_val_acc":    round(max(history["val_acc"]), 4),
        "config": config,
    }

    # ── Append to sessions log ───────────────────────────────────────────────
    log_path = os.path.join(RUNS_DIR, "sessions_log.json")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            sessions_log = json.load(f)
    else:
        sessions_log = []

    sessions_log.append(session_info)

    with open(log_path, "w") as f:
        json.dump(sessions_log, f, indent=2)

    # ── Print diff with previous session ─────────────────────────────────────
    if len(sessions_log) >= 2:
        prev = sessions_log[-2]
        curr = sessions_log[-1]
        print("\n[*] Comparison with previous training session:")
        print(f"    {'Metric':<20} {'Previous':>12} {'Current':>12} {'Diff':>10}")
        print(f"    {'-'*54}")
        for key, label in [
            ("final_val_acc",   "Val Accuracy"),
            ("final_val_loss",  "Val Loss"),
            ("final_train_acc", "Train Accuracy"),
            ("final_train_loss","Train Loss"),
            ("best_val_acc",    "Best Val Acc"),
            ("best_val_loss",   "Best Val Loss"),
            ("epochs_run",      "Epochs Run"),
        ]:
            p = prev.get(key, 0)
            c = curr.get(key, 0)
            diff = c - p
            sign = "+" if diff >= 0 else ""
            print(f"    {label:<20} {p:>12.4f} {c:>12.4f} {sign}{diff:>9.4f}")

        # Config diff
        prev_cfg = prev.get("config", {})
        curr_cfg = curr.get("config", {})
        changed = {k: (prev_cfg.get(k), v) for k, v in curr_cfg.items()
                   if prev_cfg.get(k) != v}
        if changed:
            print(f"\n    Config changes:")
            for k, (old, new) in changed.items():
                print(f"      {k}: {old} -> {new}")
    else:
        print("\n[*] First training session recorded. Future runs will show diffs.")

    print(f"\n    Session archived to: {run_dir}")
    print(f"    Sessions log: {log_path} ({len(sessions_log)} total runs)")

    return session_info


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    print("[*] Downloading Bitext Customer Support dataset ...")
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    print(f"    Loaded {len(ds)} samples")

    # ── 2. Data analysis (   L 3: null checks,    L 5: distribution) ────────
    print("\n[*] Data Analysis ...")

    # Check for missing values (   L 3     4: isnull().sum())
    import pandas as pd
    df = pd.DataFrame(ds)
    missing = df.isnull().sum()
    print(f"    Missing values per column:\n{missing.to_string()}")

    # Class distribution (   L 5     13: distribution of target classes)
    intent_counts = df["intent"].value_counts()
    print(f"\n    Intent distribution ({len(intent_counts)} intents):")
    print(f"    Min samples: {intent_counts.min()} ({intent_counts.idxmin()})")
    print(f"    Max samples: {intent_counts.max()} ({intent_counts.idxmax()})")
    print(f"    Mean: {intent_counts.mean():.0f}, Std: {intent_counts.std():.0f}")

    # ── 3. Text preprocessing (   L 7       ) ────────────────────────────────
    print("\n[*] Preprocessing text ...")
    instructions = [preprocess_text(row["instruction"]) for row in ds]
    intents      = [row["intent"] for row in ds]

    # ── 4. Build intent -> label mapping + response map ──────────────────────
    unique_intents = sorted(set(intents))
    intent2idx     = {intent: i for i, intent in enumerate(unique_intents)}
    idx2intent     = {i: intent for intent, i in intent2idx.items()}
    num_classes    = len(unique_intents)

    # Intent -> canonical response (first response per intent)
    intent_responses: dict[str, str] = {}
    for row in ds:
        intent = row["intent"]
        if intent not in intent_responses:
            intent_responses[intent] = row["response"]

    labels = np.array([intent2idx[i] for i in intents], dtype=np.int64)

    # ── 5. Build vocab & encode (   L 7    s 18-19) ──────────────────────────
    vocab = build_vocab(instructions, VOCAB_SIZE)
    X     = encode(instructions, vocab, MAX_SEQ_LEN)
    print(f"    Vocab size: {len(vocab)}")

    # ── 6. Train / test split — 80/20 (   L 4     20) ────────────────────────
    n = len(X)
    perm = np.random.default_rng(42).permutation(n)
    split = int(n * (1 - TEST_SPLIT))

    X_train, X_test = X[perm[:split]], X[perm[split:]]
    y_train, y_test = labels[perm[:split]], labels[perm[split:]]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")

    # ── 6b. K-Fold Cross-Validation (   L 5) ──────────────────────────────────
    from sklearn.model_se   Ltion import KFold

    K_FOLDS = 5
    KFOLD_EPOCHS = 15  # fewer epochs per fold for speed
    print(f"\n[*] {K_FOLDS}-Fold Cross-Validation (   L 5) ...")

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_losses = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        # Prepare fold data
        X_fold_train = torch.from_numpy(X[train_idx])
        y_fold_train = torch.from_numpy(labels[train_idx])
        X_fold_val   = torch.from_numpy(X[val_idx])
        y_fold_val   = torch.from_numpy(labels[val_idx])

        fold_train_dl = DataLoader(
            TensorDataset(X_fold_train, y_fold_train),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        fold_val_dl = DataLoader(
            TensorDataset(X_fold_val, y_fold_val),
            batch_size=BATCH_SIZE,
        )

        # Fresh model for each fold
        fold_model = IntentLSTM(
            len(vocab), EMBED_DIM, HIDDEN_DIM, num_classes,
            num_layers=NUM_LSTM_LAYERS, dropout=DROPOUT,
        ).to(device)
        fold_optim = torch.optim.Adam(fold_model.parameters(), lr=LEARNING_RATE)
        fold_criterion = nn.CrossEntropyLoss()

        # Train for fewer epochs
        for ep in range(1, KFOLD_EPOCHS + 1):
            fold_model.train()
            for xb, yb in fold_train_dl:
                xb, yb = xb.to(device), yb.to(device)
                loss = fold_criterion(fold_model(xb), yb)
                fold_optim.zero_grad()
                loss.backward()
                fold_optim.step()

        # Evaluate fold
        fold_model.eval()
        fold_correct, fold_total, fold_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in fold_val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = fold_model(xb)
                fold_loss_sum += fold_criterion(logits, yb).item() * xb.size(0)
                fold_correct += (logits.argmax(1) == yb).sum().item()
                fold_total += xb.size(0)

        fold_acc = fold_correct / fold_total
        fold_loss = fold_loss_sum / fold_total
        fold_accuracies.append(fold_acc)
        fold_losses.append(fold_loss)
        print(f"    Fold {fold}/{K_FOLDS}:  acc={fold_acc:.4f}  loss={fold_loss:.4f}")

        del fold_model  # free memory

    cv_mean_acc = np.mean(fold_accuracies)
    cv_std_acc  = np.std(fold_accuracies)
    cv_mean_loss = np.mean(fold_losses)
    print(f"\n    CV Result:  {cv_mean_acc:.4f} ± {cv_std_acc:.4f} accuracy")
    print(f"                {cv_mean_loss:.4f} mean loss")

    # ── 7. Model, loss, optimiser (   L 6    s 14-15,    L 7    s 20-21) ─────
    print(f"\n[*] Full training on 80/20 split ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = IntentLSTM(
        len(vocab), EMBED_DIM, HIDDEN_DIM, num_classes,
        num_layers=NUM_LSTM_LAYERS, dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()     # = sparse_categorical_crossentropy (   L 7     21)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler (   L 6 improvement: reduce LR on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3,
    )

    # Early stopping (   L 6     10)
    early_stop = EarlyStopping(patience=PATIENCE)

    print(f"\n[*] Training on {device} for up to {EPOCHS} epochs "
          f"(early stopping patience={PATIENCE}) ...\n")

    # ── 8. Training loop (   L 6     16) ─────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    epochs_run = 0

    for epoch in range(1, EPOCHS + 1):
        epochs_run = epoch
        # --- Train ---
        model.train()                    #    L 6     16: model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimiser.zero_grad()        #    L 6     16: optimizer.zero_grad()
            loss.backward()              #    L 6     16: loss.backward()
            optimiser.step()             #    L 6     16: optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # --- Validate (   L 6     17: model.eval() + torch.no_grad()) ---
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss_sum += criterion(logits, yb).item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        val_loss = val_loss_sum / val_total
        val_acc  = val_correct / val_total

        # Record history for plotting (   L 6     9)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch:2d}/{EPOCHS}  "
              f"loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
              f"lr={optimiser.param_groups[0]['lr']:.1e}")

        # LR scheduler step
        scheduler.step(val_loss)

        # Early stopping check (   L 6     10)
        if early_stop.step(val_loss, model):
            print(f"\n  >> Early stopping triggered at epoch {epoch}")
            early_stop.restore(model)
            break

    # ── 9. Plot training vs validation loss/accuracy (   L 6     9) ───────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"],   label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"],   label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training vs Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n    Training curves saved to {plot_path}")

    # ── 10. Classification report & confusion matrix (   L 4,    L 5) ───────────
    from sklearn.metrics import classification_report, confusion_matrix

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            preds = model(xb).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    target_names = [idx2intent[i] for i in range(num_classes)]

    cls_report = classification_report(all_labels, all_preds, target_names=target_names)
    print("\n[*] Classification Report (   L 4):\n")
    print(cls_report)

    # Save confusion matrix (   L 5)
    cm = confusion_matrix(all_labels, all_preds)
    fig_cm, ax_cm = plt.subplots(figsize=(14, 12))
    im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    plt.colorbar(im, ax=ax_cm)
    plt.tight_layout()
    cm_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"    Confusion matrix saved to {cm_path}")

    # ── 11. Save all artifacts ────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "chatbot_rnn.pth"))

    with open(os.path.join(SAVE_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(SAVE_DIR, "intent2idx.json"), "w") as f:
        json.dump(intent2idx, f)
    with open(os.path.join(SAVE_DIR, "idx2intent.json"), "w") as f:
        json.dump(idx2intent, f)
    with open(os.path.join(SAVE_DIR, "intent_responses.json"), "w") as f:
        json.dump(intent_responses, f)

    run_config = {
        "vocab_size": len(vocab),
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_classes": num_classes,
        "max_seq_len": MAX_SEQ_LEN,
        "num_lstm_layers": NUM_LSTM_LAYERS,
        "dropout": DROPOUT,
        "max_epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "test_split": TEST_SPLIT,
        "patience": PATIENCE,
        "dataset_size": len(ds),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "cv_folds": K_FOLDS,
        "cv_mean_acc": round(cv_mean_acc, 4),
        "cv_std_acc": round(cv_std_acc, 4),
        "cv_fold_accuracies": [round(a, 4) for a in fold_accuracies],
    }
    with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    with open(os.path.join(SAVE_DIR, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"\n[OK] All artifacts saved to {SAVE_DIR}/")
    print(f"     Model params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"     Vocab size   : {len(vocab):,}")
    print(f"     Intents      : {num_classes}")

    # ── 12. Archive this training session ─────────────────────────────────────
    archive_session(
        history=history,
        config=run_config,
        final_val_acc=val_acc,
        final_val_loss=val_loss,
        epochs_run=epochs_run,
        classification_rpt=cls_report,
    )


if __name__ == "__main__":
    main()
