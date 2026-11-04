# load triton before any test module imports tensorflow: both carry llvm, and loading triton second aborts
# the process (see src/torch_model.py)
try:
    import triton  # noqa: F401
except ImportError:
    pass

from types import SimpleNamespace

import transformers


class OfflineTokenizer(SimpleNamespace):
    # stand-in for the gpt-neox tokenizer, which src/tfds/common_pile_grouped.py downloads at import:
    # one token per character, so the tests run without network access
    eos_token_id = 0
    eos_token = "\0"
    padding_side = "right"

    def encode(self, s, **kwargs):
        return [1 + (ord(c) % 60) for c in s]


transformers.AutoTokenizer.from_pretrained = lambda *a, **k: OfflineTokenizer()
