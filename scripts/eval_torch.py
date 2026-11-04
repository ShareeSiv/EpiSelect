# offline eval of PyTorch runs (scripts/train_torch.py), the counterpart of src/eval.py for orbax runs:
# the common pile val set, the OOD token streams and the lm eval harness tasks of Table 1, from configs/eval.yaml
#   python scripts/eval_torch.py 'runs=[{label: EpiSelect_own, path: /data/runs/124M_EpiSelect_own}]' \
#     data_dir=/data/commonpile_tfds
# harness scoring follows src/harness/lm_harness_eval.py: zero-shot loglikelihoods, right-padded batches,
# context + continuation truncated to max_length, and acc (acc_norm only where a task has no acc)
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# isort: split
from src.torch_model import GPT, seq_losses  # noqa: E402  # first: loads triton before tensorflow imports

# isort: split

import glob  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from lm_eval.api.model import LM  # noqa: E402
from lm_eval.evaluator import simple_evaluate  # noqa: E402
from lm_eval.models.utils import Collator  # noqa: E402
from lm_eval.tasks import TaskManager  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from src.dataloader import commonpile_loader  # noqa: E402
from src.eval import OOD_SOURCES, _get_ood_tokens  # noqa: E402
from src.torch_train import evaluate  # noqa: E402


def load_run(rundir, device):
    run_cfg = OmegaConf.create(json.load(open(os.path.join(rundir, "cfg.json"))))
    path = os.path.join(rundir, "model.pt")
    if os.path.exists(path):
        state, step = torch.load(path, map_location=device), run_cfg.max_steps
    else:  # an unfinished run: its latest checkpoint
        ckpts = sorted(glob.glob(os.path.join(rundir, "ckpt_*.pt")), key=lambda p: int(os.path.basename(p)[5:-3]))
        assert ckpts, f"no model.pt or ckpt_*.pt in {rundir}"
        ckpt = torch.load(ckpts[-1], map_location=device)
        state, step = ckpt["model"], ckpt["step"] + 1
    model = GPT(run_cfg.model).to(device)
    model.load_state_dict(state)
    return model.eval(), run_cfg, int(step)


class TorchLM(LM):
    _DEFAULT_MAX_LENGTH = 2048  # as src/harness/lm_harness_eval.LMEvaluator

    def __init__(self, model, device, batch_size=32, dtype=torch.bfloat16):
        super().__init__()
        self.model, self.device, self.batch_size, self.dtype = model, device, batch_size, dtype
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    @property
    def max_length(self):
        n = getattr(self.tokenizer, "model_max_length", None)
        if n is None or n == 1000000000000000019884624838656:
            return self._DEFAULT_MAX_LENGTH
        return n

    def tok_encode(self, string):
        return self.tokenizer.encode(string)

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        return context_enc, whole_enc[len(context_enc) :]

    def loglikelihood(self, requests):
        reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                raise ValueError("Context cannot be empty.")
            reqs.append(((context, continuation), *self._encode_pair(context, continuation)))
        return self._loglikelihood_tokens(reqs)

    @torch.no_grad()
    def _loglikelihood_tokens(self, requests):
        res = []
        re_ord = Collator(requests, sort_fn=lambda x: (-len(x[1] + x[2]), tuple(x[1] + x[2])))
        pbar = tqdm(total=len(requests))
        for chunk in re_ord.get_batched(n=self.batch_size):
            inps, ctxlens = [], []
            for _, context_enc, continuation_enc in chunk:
                assert len(context_enc) > 0 and 0 < len(continuation_enc) <= self.max_length
                inps.append((context_enc + continuation_enc)[-(self.max_length + 1) :])
                ctxlens.append(len(context_enc))
            T = max(len(i) for i in inps)
            batch = torch.zeros((len(inps), T), dtype=torch.long)
            for r, inp in enumerate(inps):
                batch[r, : len(inp)] = torch.tensor(inp)
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
                logits = self.model(batch.to(self.device))
            logprobs = F.log_softmax(logits.float(), dim=-1).cpu()
            for r, (inp, ctxlen) in enumerate(zip(inps, ctxlens)):
                inplen = len(inp)
                # as the jax evaluator: ctxlen is the untruncated context length
                lp = logprobs[r, ctxlen - 1 : inplen - 1]
                cont = torch.tensor(inp[ctxlen:inplen])
                token_lp = lp.gather(-1, cont[:, None]).squeeze(-1)
                res.append((float(token_lp.sum()), bool((lp.argmax(-1) == cont).all())))
                pbar.update(1)
        pbar.close()
        return re_ord.get_original(res)

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        raise NotImplementedError("Table 1's tasks are all loglikelihood-scored")


def eval_harness(model, device, tasks, batch_size, limit):
    out = {}
    for task in tasks:
        tic = time.time()
        results = simple_evaluate(
            model=TorchLM(model, device, batch_size=batch_size),
            tasks=[task],
            num_fewshot=0,
            task_manager=TaskManager(),
            limit=limit,
            confirm_run_unsafe_code=True,
        )
        r = results["results"].get(task, {})
        acc = r.get("acc,none", r.get("acc_norm,none"))
        stderr = r.get("acc_stderr,none", r.get("acc_norm_stderr,none"))
        print(f"{task} done in {time.time() - tic:.1f}s: acc {acc:.3f}")
        out[task] = {"acc": acc, "stderr": stderr}
    out["average"] = float(np.mean([v["acc"] for v in out.values()]))
    return out


@torch.no_grad()
def eval_tokens(model, tokens, block_size, batch_size, device):
    sl = block_size + 1
    seqs = torch.from_numpy(tokens[: (len(tokens) // sl) * sl].reshape(-1, sl).astype(np.int64))
    losses = []
    for i in range(len(seqs) // batch_size):
        s = seqs[i * batch_size : (i + 1) * batch_size].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            losses.append(float(seq_losses(model, s[:, :-1], s[:, 1:]).mean()))
    loss = float(np.mean(losses))
    return {"loss": loss, "perplexity": float(np.exp(loss))}


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evals, tasks = set(cfg.evals), list(cfg.harness_tasks)
    results = {"evals": sorted(evals), "runs": {}}
    if os.path.exists(cfg.output):
        results["runs"].update(json.load(open(cfg.output)).get("runs", {}))

    for run in cfg.runs:
        key = run.label.lower().replace(" ", "_").replace("-", "_")
        print(f"\n{run.label}: {run.path}")
        if key in results["runs"]:
            print("  already done, skipping")
            continue
        model, run_cfg, step = load_run(run.path, device)
        rd = {"label": run.label, "rundir": run.path, "step": step}

        if "val" in evals:
            bs = run_cfg.get("eval_batch_size") or run_cfg.batch_size
            dset = commonpile_loader.make_eval_dset(bs, run_cfg.block_size, data_dir=cfg.data_dir)

            def loss_fn(x, y):
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    return seq_losses(model, x, y)

            _, _, m = evaluate(loss_fn, dset, run_cfg.get("eval_microbatch") or bs, device)
            rd["val"] = {k: float(v) for k, v in m.items()}
            print(f"  val balanced loss {rd['val']['balanced_loss']:.4f}")
        for name in OOD_SOURCES:
            if name in evals:
                tokens = _get_ood_tokens(name, cfg.ood_cache_dir)
                rd[name] = eval_tokens(model, tokens, run_cfg.block_size, cfg.ood_batch_size, device)
        if "harness" in evals:
            t0 = time.time()
            rd["harness"] = eval_harness(model, device, tasks, cfg.batch_size, cfg.harness_limit)
            rd["harness_time"] = time.time() - t0
            print(f"  harness average {rd['harness']['average']:.3f} (Table 1, 124M EpiSelect: 0.394)")

        results["runs"][key] = rd
        json.dump(results, open(cfg.output, "w"), indent=2)
        print(f"  saved -> {cfg.output}")


if __name__ == "__main__":
    main()
