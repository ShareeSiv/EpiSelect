import dataclasses
import itertools

import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import load_dataset

from src.common_pile_domains import (  # noqa: F401  (re-exported)
    DOMAIN_STATS,
    HF_NAMES,
    TASKS,
    TOKEN_CAP,
    VAL_FRACTION,
    get_tokenizer,
    split_rows,
)


@dataclasses.dataclass
class CommonPileConfig(tfds.core.BuilderConfig):
    hf_name: str = ""


_CONFIGS = [
    CommonPileConfig(name=task, hf_name=HF_NAMES[task], description=task)
    for task in TASKS
]


def __getattr__(name):
    # `common_pile_grouped.tokenizer` loads lazily (PEP 562)
    if name == "tokenizer":
        return get_tokenizer()
    raise AttributeError(name)


class CommonPileGrouped(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("2.0.0")  # 2.0.0: per-domain token cap
    BUILDER_CONFIGS = _CONFIGS

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                "tokens": tfds.features.Tensor(shape=(None,), dtype=tf.int32),
            }),
            supervised_keys=("tokens", "tokens"),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        hf_name = self.builder_config.hf_name
        val_rows, train_rows = split_rows(self.builder_config.name)
        return [
            tfds.core.SplitGenerator(
                name="val",
                gen_kwargs={"hf_name": hf_name, "start": 0, "stop": val_rows},
            ),
            tfds.core.SplitGenerator(
                name="train",
                gen_kwargs={"hf_name": hf_name, "start": val_rows, "stop": val_rows + train_rows},
            ),
        ]

    def _generate_examples(self, hf_name: str, start: int, stop: int):
        # streaming: fetches only the needed prefix, avoids Arrow schema errors
        ds = load_dataset(hf_name, split="train", streaming=True)
        for i, doc in enumerate(itertools.islice(ds, start, stop)):
            text = doc.get("text", "")
            if text:
                yield i, {"tokens": get_tokenizer().encode(text)}
