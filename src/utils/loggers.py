import logging
import math
import os
from importlib.util import find_spec
from typing import List, Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def index_to_multiindex(
    index: pd.Index, sep: str, max_levels: int, replace: Optional[Tuple[str, str]] = None
) -> pd.MultiIndex:
    if replace is None:
        # dummy replace
        replace = ("x", "x")
    multi_index_str = [
        c.replace(replace[0], replace[1]).split(sep, maxsplit=max_levels - 1) for c in index
    ]
    # same length
    multi_index_str = [levels + [None] * (max_levels - len(levels)) for levels in multi_index_str]
    names = index.name.split(sep) if index.name is not None else None
    return pd.MultiIndex.from_tuples(tuples=multi_index_str, names=names)


def multiindex_to_index(multiindex: pd.MultiIndex, sep: str) -> pd.Index:
    index_str = [
        sep.join([v for v in values if not (isinstance(v, float) and math.isnan(v))])
        for values in multiindex
    ]
    name = sep.join([name for name in multiindex.names if name is not None]) or None
    return pd.Index(index_str, name=name)


def load_csv_run(
    path: str, metric_prefix_whitelist: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    # get metrics
    metric_df = pd.read_csv(os.path.join(path, "metrics.csv"))
    if metric_prefix_whitelist is not None:
        if not isinstance(metric_prefix_whitelist, (list, tuple)):
            metric_prefix_whitelist = [metric_prefix_whitelist]
        cols = [
            col
            for col in metric_df.columns
            if any(col.startswith(prefix) for prefix in metric_prefix_whitelist)
        ]
        metric_df = metric_df[cols]
        # drop empty rows (e.g. train metrics when we filtered with "test/")
        metric_df = metric_df.dropna()
    if len(metric_df.columns) == 0:
        logger.warning(f"no metric data available after filtering. path={path}")
        return None
    metric_df.columns = index_to_multiindex(
        metric_df.columns, replace=("-", "/"), sep="/", max_levels=3
    )

    # get hyperparameters
    with open(os.path.join(path, "hparams.yaml")) as f:
        hparams = yaml.safe_load(f)
    hparams_df = pd.json_normalize(hparams, sep="/")
    hparams_df.columns = index_to_multiindex(hparams_df.columns, sep="/", max_levels=3)

    # combine
    # repeat to create a row for each row in metrics
    hparams_repeated = pd.concat([hparams_df] * len(metric_df), axis="index", ignore_index=True)
    # set index to join correctly (we can not use ignore_index=True because we want to keep the column labels)
    hparams_repeated.index = metric_df.index
    combined = pd.concat([metric_df, hparams_repeated], axis=1, keys=["metrics", "hparams"])
    combined.index.name = "entry"

    return combined


def load_csv_experiment(
    path: str, reduce_index_levels: bool = False, **kwargs
) -> Optional[pd.DataFrame]:
    if not os.path.isdir(path):
        raise ValueError(f"experiment path={path} does not point to a directory")

    subdirs = os.listdir(path)
    data_dict_with_empty_entries = {
        subdir: load_csv_run(path=os.path.join(path, subdir), **kwargs)
        for subdir in subdirs
        if os.path.isdir(os.path.join(path, subdir))
    }
    data_dict = {k: v for k, v in data_dict_with_empty_entries.items() if v is not None}
    if len(data_dict) == 0:
        logger.warning(f"no experiment data found in path={path}")
        return None
    run_index_names = list(data_dict.values())[0].index.names
    combined = pd.concat(
        data_dict.values(), keys=data_dict.keys(), names=["run"] + run_index_names
    )
    if reduce_index_levels:
        if len(data_dict) != len(combined):
            with_multiple_entries = [
                run_id for run_id, run_data in data_dict.items() if len(run_data) > 1
            ]
            raise Exception(
                f"can not reduce index levels, because there are multiple entries for a some runs: "
                f"{with_multiple_entries}"
            )
        combined.index = combined.index.droplevel(run_index_names)
    return combined


def load_csv_data(path: str, **kwargs) -> Optional[pd.DataFrame]:
    if not os.path.isdir(path):
        raise ValueError(f"path={path} does not point to a directory")

    subdirs = os.listdir(path)
    data_dict_with_empty_entries = {
        subdir: load_csv_experiment(path=os.path.join(path, subdir), **kwargs)
        for subdir in subdirs
        if os.path.isdir(os.path.join(path, subdir))
    }
    data_dict = {k: v for k, v in data_dict_with_empty_entries.items() if v is not None}
    if len(data_dict) == 0:
        logger.warning(f"no csv data found in path={path}")
        return None
    run_index_names = list(data_dict.values())[0].index.names
    combined = pd.concat(
        data_dict.values(), keys=data_dict.keys(), names=["experiment"] + run_index_names
    )
    return combined


if __name__ == "__main__":
    # result = load_csv_run(
    #    path="logs/logger/csv/my-experiment/version_0",
    #    #metric_prefix_whitelist=["test/"]
    # )

    # result = load_csv_experiment(
    #    path="logs/logger/csv/my-experiment",
    #    #metric_prefix_whitelist=["test/"],
    # )

    result = load_csv_data(
        path="logs/logger/csv",
        metric_prefix_whitelist="test/",
        reduce_index_levels=True,
    )

    # Show bar plot for F1 values of a certain experiment (folder in the csv output directory),
    # here "my-experiment".
    #
    # Note: the following may be required if running via PyCharm:
    # import matplotlib as mpl
    # mpl.use('TkAgg')

    if result is None:
        raise ValueError("result does not contain any entries")

    experiment_name = "my-experiment"

    # select the subset of relevant data
    data_selected = result.xs(
        key=experiment_name,
        level="experiment",
    )
    # bring the data into the required format for bar plotting
    data_plot = data_selected[("metrics", "test", "f1")].T
    # replace the columns with some hyperparameters for better readability, here dataset.select_n.stop
    data_plot.columns = data_selected[("hparams", "dataset", "select_n", "stop")]
    # rename the column index name, this will be used as caption for the legend
    data_plot.columns.name = "number of training documents"
    # plot
    # use plotly, if available
    if find_spec("plotly"):
        pd.options.plotting.backend = "plotly"
        fig = data_plot.plot.bar(title=experiment_name.replace("-", "\n"), barmode="group")
        fig.show()
    else:
        import matplotlib.pyplot as plt

        fig = data_plot.plot.bar(title=experiment_name.replace("-", "\n"))
        plt.show()

    print("done")
