import argparse

import tensorflow_datasets as tfds

from src.tfds import common_pile_grouped


def build(domain, data_dir):
    print(f"Building common_pile_grouped/{domain} -> {data_dir}", flush=True)
    builder = tfds.builder(f"common_pile_grouped/{domain}", data_dir=data_dir)
    builder.download_and_prepare()
    splits = builder.info.splits
    print(f"Done: train={splits['train'].num_examples:,} docs, val={splits['val'].num_examples:,} docs", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", nargs="?", help=f"one of: {', '.join(common_pile_grouped.TASKS)}")
    parser.add_argument("--all", action="store_true", help="build every domain sequentially")
    parser.add_argument("--data_dir", required=True, help="local path or gs:// bucket")
    args = parser.parse_args()

    if args.all:
        domains = common_pile_grouped.TASKS
    else:
        assert args.domain in common_pile_grouped.TASKS, (
            f"Unknown domain: {args.domain}. Valid: {common_pile_grouped.TASKS}"
        )
        domains = [args.domain]

    for domain in domains:
        build(domain, args.data_dir)


if __name__ == "__main__":
    main()
