import argparse

import numpy as np

from src.scaling_law_bayes import fit_bayes, summarize


def main():
    parser = argparse.ArgumentParser(description="full-bayes fit of the scaling law on a dumped (D, L) npz")
    parser.add_argument("data", help="npz with D [K,T] and L [K,T], as written by the selector's fit_data_path")
    parser.add_argument("--out", default=None, help="where to save the posterior samples (npz)")
    parser.add_argument("--num_warmup", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--num_chains", type=int, default=2)
    parser.add_argument("--gamma_scale", type=float, default=1.0, help="half-normal prior scale on gamma")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.data)
    samples = fit_bayes(
        data["D"].astype(np.float32),
        data["L"].astype(np.float32),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        seed=args.seed,
        gamma_scale=args.gamma_scale,
    )
    summarize(samples)
    if args.out:
        np.savez(args.out, **samples)


if __name__ == "__main__":
    main()
