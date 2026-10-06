"""Repair incomplete spikevision HDF5 caches from extracted AEDAT files."""
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
from snntorch.spikevision.spikedata.dvs_gesture import aedat_to_events, gather_aedat
from tqdm import tqdm


def prepare_cache(root):
    root = Path(root)
    destination = root / "dvs_gesture.hdf5"
    if destination.exists():
        with h5py.File(destination, "r") as cache:
            if "data" in cache and "extra" in cache and len(cache["data"]):
                if all(all(name in sample for name in ("times", "addrs", "labels"))
                       for sample in cache["data"].values()):
                    return

    extracted = root / "DvsGesture"
    train = sorted(gather_aedat(str(root), str(extracted), 1, 24))
    test = sorted(gather_aedat(str(root), str(extracted), 24, 30))
    if len(train) != 98 or len(test) != 24:
        raise RuntimeError(f"Expected 122 extracted AEDAT files in {extracted}")

    print("Rebuilding DVS Gesture HDF5 from local AEDAT files...")
    # Replace the cache only after a successful conversion.
    with tempfile.NamedTemporaryFile(dir=root, suffix=".hdf5", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with h5py.File(temporary, "w") as cache:
            data_group = cache.create_group("data")
            extra = cache.create_group("extra")
            keys = {True: [], False: []}
            for filename in tqdm(train + test):
                events, labels = aedat_to_events(filename)
                if np.any(events[1:, 0] < events[:-1, 0]):
                    events = events[np.argsort(events[:, 0], kind="stable")]
                subject, light = Path(filename).stem.split("_")[:2]
                is_train = filename in train
                for label, start, end in labels:
                    begin, stop = np.searchsorted(events[:, 0], [start, end])
                    gesture = events[begin:stop]
                    if not len(gesture):
                        raise ValueError(f"Empty gesture in {filename}: {start}-{end}")
                    if not 1 <= int(label) <= 11:
                        raise ValueError(f"Invalid gesture label: {label}")
                    key = len(data_group)
                    sample = data_group.create_group(str(key))
                    sample.create_dataset("times", data=gesture[:, 0] - gesture[0, 0], dtype="uint32")
                    sample.create_dataset("addrs", data=gesture[:, 1:], dtype="uint8")
                    sample.create_dataset("labels", data=int(label) - 1, dtype="uint8")
                    sample.attrs["meta_info"] = str({
                        "key": str(key), "subject": subject,
                        "light condition": light, "training sample": is_train,
                    })
                    keys[is_train].append(key)
            extra.create_dataset("train_keys", data=keys[True])
            extra.create_dataset("test_keys", data=keys[False])
            extra.attrs["Ntrain"] = len(keys[True])
            extra.attrs["Ntest"] = len(keys[False])
            extra.attrs["N"] = len(data_group)
        if destination.exists():
            backup = destination.with_suffix(".hdf5.incomplete")
            if not backup.exists():
                shutil.copy2(destination, backup)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
