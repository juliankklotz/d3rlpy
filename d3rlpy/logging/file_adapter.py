import json
import os
from enum import Enum, IntEnum
from typing import Any

import numpy as np

from .logger import (
    LOG,
    AlgProtocol,
    LoggerAdapter,
    LoggerAdapterFactory,
    SaveProtocol,
)

__all__ = ["FileAdapter", "FileAdapterFactory","UnifiedFileAdapterFactory", "UnifiedFileAdapter", "LightweightFileAdapterFactory", "LightweightFileAdapter"]


# default json encoder for numpy objects
def default_json_encoder(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (Enum, IntEnum)):
        return obj.value
    raise ValueError(f"invalid object type: {type(obj)}")


class FileAdapter(LoggerAdapter):
    r"""FileAdapter class.

    This class saves metrics as CSV files, hyperparameters as json file and
    models as d3 files.

    Args:
        algo: Algorithm.
        logdir (str): Log directory.
    """

    _algo: AlgProtocol
    _logdir: str
    _is_model_watched: bool

    def __init__(self, algo: AlgProtocol, logdir: str):
        self._algo = algo
        self._logdir = logdir
        self._is_model_watched = False
        if not os.path.exists(self._logdir):
            os.makedirs(self._logdir)
            LOG.info(f"Directory is created at {self._logdir}")

    def write_params(self, params: dict[str, Any]) -> None:
        # save dictionary as json file
        params_path = os.path.join(self._logdir, "params.json")
        with open(params_path, "w") as f:
            json_str = json.dumps(
                params, default=default_json_encoder, indent=2
            )
            f.write(json_str)

    def before_write_metric(self, epoch: int, step: int) -> None:
        pass

    def write_metric(
        self, epoch: int, step: int, name: str, value: float
    ) -> None:
        path = os.path.join(self._logdir, f"{name}.csv")
        with open(path, "a") as f:
            print(f"{epoch},{step},{value}", file=f)

    def after_write_metric(self, epoch: int, step: int) -> None:
        pass

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        # save entire model
        model_path = os.path.join(self._logdir, f"model_{epoch}.d3")
        algo.save(model_path)
        LOG.info(f"Model parameters are saved to {model_path}")

    def close(self) -> None:
        pass

    @property
    def logdir(self) -> str:
        return self._logdir

    def watch_model(
        self,
        epoch: int,
        step: int,
    ) -> None:
        assert self._algo.impl

        # write header at the first call
        if not self._is_model_watched:
            self._is_model_watched = True
            for name, grad in self._algo.impl.modules.get_gradients():
                path = os.path.join(self._logdir, f"{name}_grad.csv")
                with open(path, "w") as f:
                    print(
                        ",".join(
                            ["epoch", "step", "min", "max", "mean", "std"]
                        ),
                        file=f,
                    )

        for name, grad in self._algo.impl.modules.get_gradients():
            path = os.path.join(self._logdir, f"{name}_grad.csv")
            with open(path, "a") as f:
                min_grad = grad.min()
                max_grad = grad.max()
                mean = grad.mean()
                std = grad.std()
                print(
                    f"{epoch},{step},{min_grad},{max_grad},{mean},{std}",
                    file=f,
                )


class FileAdapterFactory(LoggerAdapterFactory):
    r"""FileAdapterFactory class.

    This class instantiates ``FileAdapter`` object.
    Log directory will be created at ``<root_dir>/<experiment_name>``.

    Args:
        root_dir (str): Top-level log directory.
    """

    _root_dir: str

    def __init__(self, root_dir: str = "d3rlpy_logs"):
        self._root_dir = root_dir

    def create(
        self, algo: AlgProtocol, experiment_name: str, n_steps_per_epoch: int
    ) -> FileAdapter:
        logdir = os.path.join(self._root_dir, experiment_name)
        return FileAdapter(algo, logdir)



class LightweightFileAdapter(FileAdapter):
    def watch_model(self, epoch: int, step: int) -> None:
        pass  # disable all *_grad.csv logging

class LightweightFileAdapterFactory(FileAdapterFactory):
    def create(
    self, algo: AlgProtocol, experiment_name: str, n_steps_per_epoch: int
    ) -> FileAdapter:
        logdir = os.path.join(self._root_dir, experiment_name)
        return LightweightFileAdapter(algo, logdir)


class UnifiedFileAdapter(FileAdapter):
    def __init__(self, algo: AlgProtocol, logdir: str):
        super().__init__(algo, logdir)
        self._metric_cache = {}  # maps (epoch, step) -> {metric_name: value}
        self._metric_keys = set()  # collect all metric names
        self._metrics_file = os.path.join(logdir, "metrics.csv")

        # Initialize header
        if not os.path.exists(self._metrics_file):
            with open(self._metrics_file, "w") as f:
                print("epoch,step", file=f)

    def write_metric(self, epoch: int, step: int, name: str, value: float) -> None:
        key = (epoch, step)
        if key not in self._metric_cache:
            self._metric_cache[key] = {"epoch": epoch, "step": step}
        self._metric_cache[key][name] = value
        self._metric_keys.add(name)

    def after_write_metric(self, epoch: int, step: int) -> None:
        # Write one row to metrics.csv after all metrics are ready
        key = (epoch, step)
        metrics = self._metric_cache.pop(key)
        all_keys = ["epoch", "step"] + sorted(self._metric_keys)

        # If header is only epoch,step — update it now
        if os.path.getsize(self._metrics_file) < 20:  # only header
            with open(self._metrics_file, "w") as f:
                print(",".join(all_keys), file=f)

        row = [str(metrics.get(k, "")) for k in all_keys]
        with open(self._metrics_file, "a") as f:
            print(",".join(row), file=f)
    
    def watch_model(self, epoch: int, step: int) -> None:
        pass  # disable all *_grad.csv logging

class UnifiedFileAdapterFactory(FileAdapterFactory):
    def create(
    self, algo: AlgProtocol, experiment_name: str, n_steps_per_epoch: int
    ) -> FileAdapter:
        logdir = os.path.join(self._root_dir, experiment_name)
        return UnifiedFileAdapter(algo, logdir)