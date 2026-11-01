"""Build the compact Common Pile corpus: one flat uint16 token stream per domain and split.

    python scripts/build_compact_commonpile.py --out commonpile_compact --cap 1e9 --workers 16
    python scripts/build_compact_commonpile.py --out part1 --domains github stackexchange --cap 1e9

Layout (read by src/dataloader/compact_loader.py):

    <out>/meta.json                       tokenizer, eos id, cap, domains built
    <out>/<domain>/train.bin, val.bin     uint16 GPT-NeoX tokens, every doc followed by EOS
    <out>/<domain>/meta.json              doc and token counts per split

Same domains, HF sources, document split and tokenizer as the tfds build (src/tfds/common_pile_grouped.py),
with the per-domain cap as a flag: --cap 1e9 is 23.3B tokens (~47 GB), --cap 2e9 the repo's 43.3B (~87 GB).
--synthetic_tokens N writes bigram stand-in domains instead (no network, no tokenizer; src/dataloader/synthetic_loader.py),
with the real domain names, for smoke runs and for benchmarking a Kaggle TPU before the real data is uploaded.

A domain with meta.json is complete and skipped on rerun, so the build can run domain by domain on a small disk
(build some --domains, upload them as one Kaggle dataset, delete, continue). The loader finds domains anywhere
under its data_dir, so several Kaggle datasets mounted under /kaggle/input work together.
"""

import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from src.common_pile_domains import DOMAIN_STATS, HF_NAMES, TASKS, TOKENIZER_NAME, get_tokenizer, split_rows

MAX_TOKEN_ID = np.iinfo(np.uint16).max


def _encode(texts):
    tok = get_tokenizer()
    eos = tok.eos_token_id
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    out = np.fromiter(itertools.chain.from_iterable(x + [eos] for x in ids), dtype=np.int64)
    assert out.size == 0 or out.max() <= MAX_TOKEN_ID, "token id does not fit in uint16"
    return out.astype(np.uint16), len(ids)


def _chunks(docs, chunk_docs):
    while True:
        texts = [d.get("text", "") for d in itertools.islice(docs, chunk_docs)]
        if not texts:
            return
        texts = [t for t in texts if t]  # same as the tfds build: empty docs are dropped
        if texts:
            yield texts


def _write_split(path, docs, pool, chunk_docs, log_every=50_000_000):
    tmp = path + ".tmp"
    n_tok = n_doc = 0
    next_log, t0 = log_every, time.time()
    with open(tmp, "wb") as f:
        for arr, n in pool.imap(_encode, _chunks(docs, chunk_docs)):
            arr.tofile(f)
            n_tok += arr.size
            n_doc += n
            if n_tok >= next_log:
                print(f"  {os.path.basename(path)}: {n_tok / 1e6:,.0f}M tokens, {n_tok / (time.time() - t0) / 1e6:.2f}M tok/s", flush=True)
                next_log += log_every
    os.replace(tmp, path)
    return n_tok, n_doc


def build_domain(domain, out, cap, pool, chunk_docs, dataset_fn=None):
    ddir = os.path.join(out, domain)
    meta_path = os.path.join(ddir, "meta.json")
    if os.path.exists(meta_path):
        print(f"{domain}: already built, skipping", flush=True)
        return json.load(open(meta_path))
    os.makedirs(ddir, exist_ok=True)
    val_rows, train_rows = split_rows(domain, token_cap=cap)
    print(f"{domain}: {val_rows:,} val + {train_rows:,} train docs from {HF_NAMES[domain]}", flush=True)

    if dataset_fn is None:
        from datasets import load_dataset

        def dataset_fn(name):
            return load_dataset(name, split="train", streaming=True)

    # one pass over the stream: the first val_rows docs are val, the next train_rows are train (as the tfds build)
    docs = iter(dataset_fn(HF_NAMES[domain]))
    meta = {"domain": domain, "hf_name": HF_NAMES[domain], "cap": cap}
    for split, rows in (("val", val_rows), ("train", train_rows)):
        n_tok, n_doc = _write_split(os.path.join(ddir, f"{split}.bin"), itertools.islice(docs, rows), pool, chunk_docs)
        meta[split] = {"tokens": n_tok, "docs": n_doc}
        print(f"{domain}/{split}: {n_doc:,} docs, {n_tok:,} tokens", flush=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def build_synthetic(domains, out, tokens, vocab=256, seed=0):
    # domain i gets tokens * (2 - i / (K - 1)) train tokens (unequal, so the natural mixture is not uniform), 2% val
    from src.dataloader.synthetic_loader import SyntheticTasksSampler

    K, window = len(domains), 1024
    for split, frac in (("train", 1.0), ("val", 0.02)):
        sampler = SyntheticTasksSampler(split, window - 1, tasks=domains, vocab=vocab, seed=seed)
        for i, domain in enumerate(domains):
            n_tok = int(tokens * frac * (2.0 - i / max(K - 1, 1)))
            n_win = max(1, -(-n_tok // window))
            ddir = os.path.join(out, domain)
            os.makedirs(ddir, exist_ok=True)
            sampler._sample(i, n_win).astype(np.uint16).reshape(-1)[:n_tok].tofile(os.path.join(ddir, f"{split}.bin"))
    for domain in domains:
        ddir = os.path.join(out, domain)
        meta = {"domain": domain, "synthetic": True, "vocab": vocab}
        for split in ("train", "val"):
            meta[split] = {"tokens": os.path.getsize(os.path.join(ddir, f"{split}.bin")) // 2, "docs": None}
        with open(os.path.join(ddir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"format": "uint16 synthetic bigram tokens", "synthetic": True, "domains": list(domains)}, f, indent=2)
    print(f"Wrote {K} synthetic domains to {out}")


def main(argv=None, dataset_fn=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cap", type=float, default=1e9, help="max tokens per domain (approximate, via mean doc length)")
    parser.add_argument("--domains", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--chunk_docs", type=int, default=256, help="docs per tokenizer call")
    parser.add_argument("--synthetic_tokens", type=float, default=0, help="write synthetic domains of ~this many tokens")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    if args.synthetic_tokens:
        return build_synthetic(args.domains, args.out, int(args.synthetic_tokens))
    built = {}
    with Pool(args.workers) as pool:
        for domain in args.domains:
            built[domain] = build_domain(domain, args.out, int(args.cap), pool, args.chunk_docs, dataset_fn)

    tok = get_tokenizer()
    top = {
        "format": "uint16 flat token stream, docs joined by eos",
        "tokenizer": TOKENIZER_NAME,
        "eos_token_id": tok.eos_token_id,
        "cap": int(args.cap),
        "domains": sorted(built),
        "train_tokens": sum(m["train"]["tokens"] for m in built.values()),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(top, f, indent=2)
    est = sum(min(n * m, args.cap) for d, (n, m) in DOMAIN_STATS.items() if d in built)
    print(f"Done: {len(built)} domains, {top['train_tokens']:,} train tokens (estimated {est:,.0f}) in {args.out}")


if __name__ == "__main__":
    main()
