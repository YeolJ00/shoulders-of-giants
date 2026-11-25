"""Load a downloaded pickle without running its code: only numpy arrays and plain containers are rebuilt, any other
global the pickle names is refused (a pickle can call anything it imports)."""
import pickle

ALLOWED = {
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "slice"), ("builtins", "complex"),
    ("collections", "OrderedDict"),
}


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def load(path):
    with open(path, "rb") as f:
        return _Restricted(f).load()
