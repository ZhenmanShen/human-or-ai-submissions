"""Frozen pretrained text-embedding model -> linear classifier.

The encoder (e.g. BAAI/bge-base-en-v1.5, intfloat/e5-base-v2) is never
trained: we embed every passage once (mean- or CLS-pooled, L2-normalised),
cache the vectors in outputs/cache/, and fit a logistic regression (or
LightGBM) on top. Cheap to run on a laptop, and a useful contrast with
fine-tuning: how much does a general-purpose semantic embedding know about
"AI-ness" without task-specific training?

device: auto | cuda | cuda:<i> | mps | cpu   dtype: auto | fp32 | fp16 | bf16
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
from tqdm.auto import tqdm

from ..config import ROOT
from ..features import stylometric_features
from .base import BaseModel


class EmbeddingExtractor:
    def __init__(self, model_name, pooling="mean", prefix="", max_length=512, batch_size=32,
                 device="auto", dtype="auto"):
        self.model_name, self.pooling, self.prefix = model_name, pooling, prefix
        self.max_length, self.batch_size, self.device_pref, self.dtype = max_length, batch_size, device, dtype
        safe = model_name.replace("/", "__").replace("\\", "__").replace(":", "_")
        self.cache_name = f"emb_{safe}_{pooling}_{max_length}_{hashlib.md5(prefix.encode()).hexdigest()[:6]}.joblib"
        self._cache = None
        self._model = None

    @property
    def cache_path(self) -> Path:
        return ROOT / "outputs" / "cache" / self.cache_name

    def _load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from ..utils import device as pick_device, device_type
        self.torch = torch
        self.device = pick_device(self.device_pref)
        d = str(self.dtype).lower()
        if d == "auto":
            dt = torch.float16 if device_type(self.device) == "cuda" else torch.float32
        else:
            dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[d]
        from ..utils import hf_source
        src = hf_source(self.model_name)
        self.tok = AutoTokenizer.from_pretrained(src)
        try:
            m = AutoModel.from_pretrained(src, dtype=dt)
        except TypeError:
            m = AutoModel.from_pretrained(src, torch_dtype=dt)
        self._model = m.to(self.device).eval()
        print(f"  embeddings: {self.model_name} on {self.device}", flush=True)

    def _compute(self, texts):
        if self._model is None:
            self._load()
        torch = self.torch
        order = np.argsort([len(t) for t in texts])[::-1]
        out = [None] * len(texts)
        with torch.inference_mode():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="embeddings", leave=False):
                idx = order[i:i + self.batch_size]
                enc = self.tok([self.prefix + texts[j] for j in idx], truncation=True, padding=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                h = self._model(**enc).last_hidden_state.float()
                if self.pooling == "cls":
                    v = h[:, 0]
                else:
                    m = enc["attention_mask"].unsqueeze(-1).float()
                    v = (h * m).sum(1) / m.sum(1).clamp(min=1)
                v = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
                for k, j in enumerate(idx):
                    out[j] = v[k]
        return out

    def transform(self, texts) -> np.ndarray:
        texts = list(texts)
        if self._cache is None:
            self._cache = joblib.load(self.cache_path) if self.cache_path.exists() else {}
        keys = [hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts]
        todo = list({k: t for k, t in zip(keys, texts) if k not in self._cache}.items())
        if todo:
            vecs = self._compute([t for _, t in todo])
            self._cache.update({k: v for (k, _), v in zip(todo, vecs)})
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(f".tmp{os.getpid()}")
            joblib.dump(self._cache, tmp)
            os.replace(tmp, self.cache_path)
        return np.stack([self._cache[k] for k in keys]).astype(np.float32)

    def release(self):
        self._model = None
        if hasattr(self, "torch"):
            import gc
            gc.collect()
            if self.device.startswith("cuda"):
                self.torch.cuda.empty_cache()

    def __getstate__(self):
        s = self.__dict__.copy()
        for k in ("_model", "tok", "torch", "device"):
            s.pop(k, None)
        s["_cache"], s["_model"], s["device_pref"] = None, None, "auto"
        return s


class EmbeddingLogReg(BaseModel):
    name = "embedding_logreg"

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", pooling: str = "cls",
                 prefix: str = "", max_length: int = 512, batch_size: int = 32, dtype: str = "auto",
                 head: str = "logreg", C: float = 1.0, add_stylometric: bool = False,
                 lgbm: dict | None = None, seed: int = 42, device: str = "auto"):
        super().__init__(model_name=model_name, pooling=pooling, head=head, C=C)
        self.extractor = EmbeddingExtractor(model_name, pooling, prefix, max_length, batch_size, device, dtype)
        self.head, self.C, self.add_stylometric = head, C, add_stylometric
        self.lgbm_params, self.seed = lgbm or {}, seed

    def set_device(self, device: str) -> None:
        self.extractor.device_pref = device
        self.extractor._model = None

    def precompute(self, texts) -> None:
        self.extractor.transform(texts)
        self.extractor.release()

    def _features(self, texts):
        F = self.extractor.transform(texts)
        if self.add_stylometric:
            F = np.hstack([F, stylometric_features(texts)])
        return F

    def fit(self, texts, y, val_texts=None, val_y=None, unlabeled_texts=None):
        X = self._features(texts)
        if self.head == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=5000))
        elif self.head == "lightgbm":
            import lightgbm as lgb
            p = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, colsample_bytree=0.5,
                     verbose=-1, random_state=self.seed)
            p.update(self.lgbm_params)
            self.clf = lgb.LGBMClassifier(**p)
        else:
            raise ValueError(f"Unknown head {self.head}")
        self.clf.fit(X, y)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._features(texts))[:, 1]
