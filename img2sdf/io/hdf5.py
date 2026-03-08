"""HDF5 serialisation/deserialisation helpers.

Merged from uSCMAN WriteData.py + ReadHDF5File.py into a single module.
"""
from __future__ import annotations
import numpy as np


def save_dict_to_hdf5(data: dict, filename: str) -> None:
    """Recursively save a nested dict of ndarrays and scalars to HDF5."""
    import h5py
    with h5py.File(filename, "w") as f:
        _write_group(f, data)


def _write_group(group, data: dict) -> None:
    import h5py
    for key, val in data.items():
        key = str(key)
        if isinstance(val, dict):
            sub = group.require_group(key)
            _write_group(sub, val)
        elif isinstance(val, np.ndarray):
            group.create_dataset(key, data=val)
        elif isinstance(val, (int, float, str, bool)):
            group.attrs[key] = val
        elif isinstance(val, list):
            group.create_dataset(key, data=np.array(val))
        else:
            try:
                group.attrs[key] = str(val)
            except Exception:
                pass


def load_dict_from_hdf5(filename: str) -> dict:
    """Recursively load an HDF5 file back into a nested dict."""
    import h5py
    with h5py.File(filename, "r") as f:
        return _read_group(f)


def _read_group(group) -> dict:
    import h5py
    out = {}
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            out[key] = _read_group(item)
        elif isinstance(item, h5py.Dataset):
            out[key] = item[()]
    for key, val in group.attrs.items():
        out[key] = val
    return out


def print_dict(d: dict, indent: int = 0) -> None:
    """Pretty-print a nested dict (mirrors uSCMAN ReadHDF5File.print_dict)."""
    prefix = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}:")
            print_dict(v, indent + 1)
        elif isinstance(v, np.ndarray):
            print(f"{prefix}{k}: ndarray {v.shape} {v.dtype}")
        else:
            print(f"{prefix}{k}: {v}")