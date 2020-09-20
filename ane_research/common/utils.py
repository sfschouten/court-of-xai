import json
import os
from os import PathLike
from typing import Generator, Iterable

from allennlp.common import JsonDict
import pandas as pd

def ensure_dir(file_path: PathLike) -> PathLike:
    if file_path[-1] != '/':
        file_path += '/'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path

def batch(iterable: Iterable, n: int = 1) -> Generator[Iterable, None, None]:
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def write_frame(
    frame: pd.DataFrame,
    base_path: PathLike,
    name: str
) -> None:
    frame.to_pickle(os.path.join(os.path.join(base_path, name + '.pkl')))
    with open(os.path.join(os.path.join(base_path, name + '.csv')), 'w+') as handle:
        frame.to_csv(handle)

def write_json(
    data: JsonDict,
    base_path: PathLike,
    name: str
) -> None:
    with open(os.path.join(os.path.join(base_path, name + '.json')), 'w+', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=4)

def model_already_trained(
    base_path: PathLike
) -> bool:
    return os.path.isfile(os.path.join(base_path, 'model.tar.gz'))
