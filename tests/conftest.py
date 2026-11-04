# load triton before any test module imports tensorflow: both carry llvm, and loading triton second aborts
# the process (see src/torch_model.py)
try:
    import triton  # noqa: F401
except ImportError:
    pass
