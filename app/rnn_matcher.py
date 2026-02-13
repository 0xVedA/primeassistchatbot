"""
rnn_matcher.py
==============
Loads the trained LSTM model and artifacts, predicts the intent
for a given user query, and returns the corresponding response.

Preprocessing must match train_rnn.py (Lec 7 cell 14 techniques).
"""

import json, os, pickle, re, string
import numpy as np
import torch
import torch.nn as nn

# ─── Attempt to import nltk (must mirror train_rnn.py) ───────────────────────
# Keep interrogative words — they carry critical intent information
KEEP_WORDS = {"where", "what", "how", "when", "which", "why", "can", "could",
              "would", "should", "want", "need", "track", "check", "cancel",
              "change", "get", "find", "status"}

try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    STOP_WORDS = set(stopwords.words("english")) - KEEP_WORDS
    STEMMER = PorterStemmer()
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    STOP_WORDS = set()
    STEMMER = None


# ─── Model definition (must match train_rnn.py) ──────────────────────────────
class IntentLSTM(nn.Module):
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
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(self.dropout(h))


# ─── Text preprocessing (must match train_rnn.py — Lec 7 cell 14) ────────────
def _preprocess(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    if HAS_NLTK:
        tokens = [STEMMER.stem(w) for w in tokens if w not in STOP_WORDS]
    else:
        basic_stops = {"the", "a", "an", "is", "are", "was", "were", "i", "my",
                       "me", "we", "our", "you", "your", "it", "its", "to", "of",
                       "in", "on", "for", "with", "and", "or", "but", "not", "do",
                       "does", "did", "have", "has", "had", "be", "been", "am"}
        tokens = [w for w in tokens if w not in basic_stops]
    return " ".join(tokens)


# ─── Public class ─────────────────────────────────────────────────────────────
class RNNMatcher:
    """Load once, call  predict(query)  to get (intent, response, confidence)."""

    def __init__(self, model_dir: str = "model"):
        # Load config
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)

        # Load mappings
        with open(os.path.join(model_dir, "vocab.pkl"), "rb") as f:
            self.vocab: dict[str, int] = pickle.load(f)
        with open(os.path.join(model_dir, "idx2intent.json")) as f:
            self.idx2intent: dict[str, str] = json.load(f)
        with open(os.path.join(model_dir, "intent_responses.json")) as f:
            self.intent_responses: dict[str, str] = json.load(f)

        self.max_seq_len = cfg["max_seq_len"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model with cfg params
        self.model = IntentLSTM(
            cfg["vocab_size"], cfg["embed_dim"], cfg["hidden_dim"],
            cfg["num_classes"],
            num_layers=cfg.get("num_lstm_layers", 2),
            dropout=cfg.get("dropout", 0.5),
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, "chatbot_rnn.pth"), map_location=self.device)
        )
        self.model.eval()

    def _encode(self, text: str) -> torch.Tensor:
        unk = self.vocab["<UNK>"]
        ids = [self.vocab.get(w, unk) for w in text.split()][:self.max_seq_len]
        ids += [0] * (self.max_seq_len - len(ids))
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def predict(self, query: str) -> tuple[str, str, float]:
        """Returns (intent, response, confidence)."""
        cleaned = _preprocess(query)
        x = self._encode(cleaned)

        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)

        intent   = self.idx2intent[str(idx.item())]
        response = self.intent_responses.get(intent, "Sorry, I don't have an answer for that.")
        return intent, response, float(conf.item())
