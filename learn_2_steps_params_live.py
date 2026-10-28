#!/usr/bin/env python3
"""
Two‑phase C2xG runner (single file, in‑script config).

Phase A: Train FastText embeddings on EMBED_CORPUS (file or directory of text files).
Phase B: Learn C2xG grammar on GRAMMAR_CORPUS (file or directory), *reusing* the embeddings from Phase A.

All settings are defined below in the "USER SETTINGS" block. No command‑line args needed.
Tested with Python 3.10, gensim 4.3.x, scikit-learn 1.7.x, pandas 2.3.x, statsmodels 0.14.x,
and a c2xg install that includes the compatibility fixes you applied earlier.
"""

from __future__ import annotations
import os, sys, time, json, math, gc, multiprocessing as mp
from pathlib import Path
from typing import Iterator, List
import re
import random
import numpy as np
from gensim.models import FastText

# Ensure we import the package from the repo checkout (c2xg/) not the parent folder shadow
ROOT = Path(__file__).resolve().parent
PKG_ROOT = ROOT / "c2xg"
if PKG_ROOT.exists():
    sys.path.insert(0, str(PKG_ROOT))

from c2xg import C2xG

# Set seeds for reproducibility
RANDOM_SEED = 42  # or whatever seed you want
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================
# USER SETTINGS (regular run)
# =========================

# Paths
EMBED_CORPUS   = Path(sys.argv[1]).resolve()   # file OR directory
GRAMMAR_CORPUS = Path(sys.argv[2]).resolve()   # file OR directory
OUT_DIR        = Path(sys.argv[3]).resolve()
NICKNAME       = "twophase.aall.17M_embed_8M_gram"

# Embeddings (FastText) — 17M tokens
E_MIN_COUNT  = 10        # keeps rare noise out; controls vocab size
E_VECTOR_SZ  = 200       # good quality/memory balance
E_WINDOW     = 5
E_EPOCHS     = 10        # bump to 15 if you want a quality nudge and can wait longer
E_MAX_WORDS  = None      # use all 17M; set to an int to pilot

# Grammar — 8M tokens total processed
G_MIN_COUNT   = 20
G_MAX_VOCAB   = 9000     # main RAM/time dial; 8k–10k is OK on 20 GB; 9k is a safe sweet spot
G_NPMI_THRES  = 0.1      # phrase mining threshold; raise to 0.2 for fewer/more reliable phrases
G_NORMALIZE   = "npmi"
G_MAX_WORDS   = 8_000_000   # <- your target corpus size for grammar
FORGET_ROUNDS = 0
INCREMENTS    = 1
GET_EXAMPLES  = True        # turn on if you want examples files

# Parallelism
N_WORKERS = min(10, mp.cpu_count())  # you reported ~10 vCPUs

# ===============
# END USER SETTINGS
# ===============


# --- small utilities ---
def _yield_text_files(root: Path) -> Iterator[Path]:
    """Yield UTF‑8 text files from a file or directory recursively."""
    if root.is_file():
        yield root
    else:
        for p in root.rglob("*"):
            if p.is_file():
                yield p

def _simple_tokenize(s: str) -> List[str]:
    # A simple, reasonably fast whitespace/punct tokenizer
    return re.findall(r"\w+(?:'\w+)?", s.lower())

def iter_corpus_lines(target: Path, max_words: int | None = None) -> Iterator[List[str]]:
    """Iterator of tokenized sentences/lines for gensim FastText/Word2Vec training.
    If max_words is set, stops after reaching that many tokens.
    """
    seen = 0
    for fp in _yield_text_files(target):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    toks = _simple_tokenize(line)
                    if not toks:
                        continue
                    seen += len(toks)
                    yield toks
                    if max_words is not None and seen >= max_words:
                        return
        except Exception as e:
            print(f"[warn] skipping {fp}: {e}", file=sys.stderr)

def _count_tokens_quick(target: Path, max_files: int = 2000) -> int:
    """Crude token count for progress/ETA reporting (samples up to max_files)."""
    n, files = 0, 0
    for fp in _yield_text_files(target):
        files += 1
        if files > max_files: break
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    n += len(_simple_tokenize(line))
        except Exception:
            pass
    return n

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def train_fasttext_pair(corpus: Path,
                        min_count: int,
                        vector_size: int,
                        window: int,
                        epochs: int,
                        max_words: int | None,
                        n_workers: int,
                        out_dir: Path,
                        nick: str):
    """Train CBOW and SG FastText models on 'corpus'; save to out_dir. Return (cbow, sg)."""
    from gensim.models import FastText

    print("\n[Embeddings] Counting tokens (sampled) …")
    approx_tokens = _count_tokens_quick(corpus)
    print(f"[Embeddings] Sampled token count (rough): {approx_tokens:,}")

    print("[Embeddings] Building training iterable …")
    data_all = list(iter_corpus_lines(corpus, max_words=max_words))
    total_examples = len(data_all)
    total_tokens = sum(len(x) for x in data_all)
    print(f"[Embeddings] Prepared {total_examples:,} lines, {total_tokens:,} tokens")

    _ensure_dir(out_dir)
    # CBOW
    print("[Embeddings] Training CBOW …")
    cbow = FastText(
        sg=0, vector_size=vector_size, window=window,
        min_count=min_count, workers=n_workers,
        hs=0, negative=100, alpha=0.01, sorted_vocab=1, min_n=3, max_n=6,
    )
    cbow.build_vocab(data_all)
    t0 = time.time()
    cbow.train(data_all, total_examples=total_examples, epochs=epochs)
    t1 = time.time()
    print(f"[Embeddings] CBOW done in {t1 - t0:.1f}s")
    cbow.save(str(out_dir / f"{nick}.embeddings.cbow.model"))
    cbow.wv.save_word2vec_format(str(out_dir / f"{nick}.embeddings.cbow.bin"), binary=True)

    # SG
    print("[Embeddings] Training Skip‑gram …")
    sg = FastText(
        sg=1, vector_size=vector_size, window=window,
        min_count=min_count, workers=n_workers,
        hs=0, negative=100, alpha=0.01, sorted_vocab=1, min_n=3, max_n=6,
    )
    sg.build_vocab(data_all)
    t0 = time.time()
    sg.train(data_all, total_examples=total_examples, epochs=epochs)
    t1 = time.time()
    print(f"[Embeddings] SG done in {t1 - t0:.1f}s")
    sg.save(str(out_dir / f"{nick}.embeddings.sg.model"))
    sg.wv.save_word2vec_format(str(out_dir / f"{nick}.embeddings.sg.bin"), binary=True)

    _save_json({
        "min_count": min_count, "vector_size": vector_size, "window": window, "epochs": epochs,
        "n_workers": n_workers, "max_words": max_words, "lines": total_examples, "tokens": total_tokens
    }, out_dir / f"{nick}.embeddings.meta.json")

    return cbow, sg

def _attach_rank_by_centrality(model):
    """Inject rank_by_centrality(model, words, use_norm=True) onto a FastText model."""
    if model is None or isinstance(model, bool):
        return

    import numpy as np
    kv = getattr(model, "wv", model)

    def rank_by_centrality(words, use_norm=True):
        vecs, kept = [], []
        for w in words:
            if hasattr(kv, "key_to_index") and w not in kv.key_to_index:
                continue
            try:
                v = kv[w].astype("float32", copy=False)
            except Exception:
                continue
            if use_norm:
                n = float(np.linalg.norm(v))
                if n:
                    v = v / n
            vecs.append(v); kept.append(w)

        if not vecs:
            return []

        mean = np.mean(np.stack(vecs, axis=0), axis=0)
        if use_norm:
            n = float(np.linalg.norm(mean))
            if n:
                mean = mean / n

        out = []
        for w in kept:
            v = kv[w].astype("float32", copy=False)
            if use_norm:
                n = float(np.linalg.norm(v))
                if n:
                    v = v / n
            out.append((float(np.dot(v, mean)), w))
        out.sort(key=lambda t: t[0], reverse=True)
        return out

    # Attach to both the model and its KeyedVectors view
    setattr(model, "rank_by_centrality", rank_by_centrality)
    if kv is not model:
        setattr(kv, "rank_by_centrality", rank_by_centrality)

def _monkey_attach_embeddings(cxg, cbow_model, sg_model):
    """Make C2xG.learn() reuse provided models instead of re-training internally,
    and ensure rank_by_centrality is available on those models."""
    # nothing to attach?
    if (cbow_model is None or isinstance(cbow_model, bool)) and \
       (sg_model   is None or isinstance(sg_model, bool)):
        print("[C2xG] No embeddings provided; leaving defaults.")
        return

    # attach models
    cxg.cbow_model = cbow_model
    cxg.sg_model   = sg_model

    # Also attach to the Loader instance
    if hasattr(cxg, 'Load'):
        cxg.Load.cbow_model = cbow_model
        cxg.Load.sg_model = sg_model

    # disable internal training
    if hasattr(cxg, "learn_embeddings"):
        def _noop(self, *a, **k):
            return True
        cxg.learn_embeddings = _noop.__get__(cxg, type(cxg))
    print("[C2xG] Attached pre-trained embeddings; internal training disabled.")

    # add rank_by_centrality where applicable
    for m in (cbow_model, sg_model):
        _attach_rank_by_centrality(m)

    # (optional) re-attach on the attributes living on cxg, just in case
    for name in ("cbow_model", "sg_model"):
        _attach_rank_by_centrality(getattr(cxg, name, None))

def learn_grammar(grammar_corpus: Path,
                  out_dir: Path,
                  nickname: str,
                  g_min_count: int,
                  g_max_vocab: int,
                  g_npmi_thres: float,
                  g_normalize: str,
                  g_max_words: int | None,
                  forget_rounds: int,
                  increments: int,
                  get_examples: bool,
                  cbow_model,
                  sg_model):
    """Run C2xG grammar learning using pre‑trained models."""

    _ensure_dir(out_dir)

    cxg = C2xG(out_dir=str(out_dir), nickname=nickname)

    # AFTER: cxg = C2xG(...); cxg.attach_pretrained_embeddings(cbow_model, sg_model)
    for m in (getattr(cxg, "cbow_model", None), getattr(cxg, "sg_model", None)):
        if m is not None:
            _attach_rank_by_centrality(m)

    cxg.normalization=g_normalize

    _monkey_attach_embeddings(cxg, cbow_model, sg_model)

    grammar_input = grammar_corpus
    tmp_file = None
    if g_max_words is not None:
        tmp_file = out_dir / f"{nickname}.grammar_sample.txt"
        print(f"[C2xG] Sampling grammar corpus to {g_max_words:,} tokens -> {tmp_file}")
        taken = 0
        with open(tmp_file, "w", encoding="utf-8") as w:
            for fp in _yield_text_files(grammar_corpus):
                with open(fp, "r", encoding="utf-8") as r:
                    for line in r:
                        toks = _simple_tokenize(line)
                        if not toks:
                            continue
                        w.write(" ".join(toks) + "\n")
                        taken += len(toks)
                        if taken >= g_max_words:
                            break
                if taken >= g_max_words:
                    break
        grammar_input = tmp_file

    print("[C2xG] Starting grammar learning …")
    t0 = time.time()
    cxg.learn(
        input_data=str(grammar_input),
        forgetting_rounds=forget_rounds,
        max_vocab=g_max_vocab,
        min_count=g_min_count,
        npmi_threshold=g_npmi_thres,
        increments=increments,
        get_examples=get_examples,
    )
    t1 = time.time()
    print(f"[C2xG] Grammar learning finished in {t1 - t0:.1f}s")

    if hasattr(cxg, "save"):
        try:
            cxg.save()
        except Exception:
            pass

    if tmp_file and tmp_file.exists():
        try:
            tmp_file.unlink()
        except Exception:
            pass

    return cxg


def main():
    if not EMBED_CORPUS.exists():
        print(f"Error: EMBED_CORPUS does not exist: {EMBED_CORPUS}", file=sys.stderr)
        sys.exit(2)
    if not GRAMMAR_CORPUS.exists():
        print(f"Error: GRAMMAR_CORPUS does not exist: {GRAMMAR_CORPUS}", file=sys.stderr)
        sys.exit(2)

    _ensure_dir(OUT_DIR)

    cbow_model_path_str = str(OUT_DIR / f"{NICKNAME}.embeddings.cbow.model")
    sg_model_path_str = str(OUT_DIR / f"{NICKNAME}.embeddings.sg.model")
    cbow_model_path = Path(cbow_model_path_str) 
    sg_model_path = Path(sg_model_path_str) 
    cbow = None
    sg = None

    if cbow_model_path.exists():
        print("Loading CBOW model")
        cbow = FastText.load(cbow_model_path_str)

    if sg_model_path.exists():
        print("Loading SG model")
        sg = FastText.load(sg_model_path_str)

    if not cbow_model_path.exists() or not sg_model_path.exists():
        cbow, sg = train_fasttext_pair(
            corpus=EMBED_CORPUS,
            min_count=E_MIN_COUNT,
            vector_size=E_VECTOR_SZ,
            window=E_WINDOW,
            epochs=E_EPOCHS,
            max_words=E_MAX_WORDS,
            n_workers=N_WORKERS,
            out_dir=OUT_DIR,
            nick=NICKNAME,
        )

    gc.collect()

    learn_grammar(
        grammar_corpus=GRAMMAR_CORPUS,
        out_dir=OUT_DIR,
        nickname=NICKNAME,
        g_min_count=G_MIN_COUNT,
        g_max_vocab=G_MAX_VOCAB,
        g_npmi_thres=G_NPMI_THRES,
        g_normalize=G_NORMALIZE,
        g_max_words=G_MAX_WORDS,
        forget_rounds=FORGET_ROUNDS,
        increments=INCREMENTS,
        get_examples=GET_EXAMPLES,
        cbow_model=cbow,
        sg_model=sg,
    )

    cfg = {
        "embed": {
            "corpus": str(EMBED_CORPUS),
            "min_count": E_MIN_COUNT, "vector_size": E_VECTOR_SZ, "window": E_WINDOW,
            "epochs": E_EPOCHS, "max_words": E_MAX_WORDS, "n_workers": N_WORKERS
        },
        "grammar": {
            "corpus": str(GRAMMAR_CORPUS),
            "min_count": G_MIN_COUNT, "max_vocab": G_MAX_VOCAB,
            "npmi_threshold": G_NPMI_THRES, "normalization": G_NORMALIZE,
            "max_words": G_MAX_WORDS, "forgetting_rounds": FORGET_ROUNDS,
            "increments": INCREMENTS, "get_examples": GET_EXAMPLES
        },
        "out_dir": str(OUT_DIR), "nickname": NICKNAME
    }
    _save_json(cfg, OUT_DIR / f"{NICKNAME}.run_config.json")
    print("\n[Done] Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
