import os
import re
from torch.utils.data import Dataset
from typing import Union
from .file import FileLineIndexer
from copy import deepcopy

class DataProvider(object):
    """Base class for data providers.

    A DataProvider holds a ``data_dict`` mapping split names (e.g.
    ``"train"``, ``"validation_copy"``) to dataset objects. Subclasses
    populate this dict from files, HuggingFace datasets, etc.
    """

    def __init__(self):
        self.data_dict = dict()

    def items(self):
        return self.data_dict.items()


class BaseDataset(Dataset):
    """Minimal Dataset base with a ``streaming`` flag (default False)."""

    def __init__(self):
        self.streaming = False


class ZipLines(BaseDataset):
    """Dataset that reads parallel lines from multiple files.

    Each field (e.g. ``"src"``, ``"trg"``) is backed by a
    :class:`FileLineIndexer` for O(1) random access by line number.

    :param config: Mapping from field name to ``{"file": path, "map": fn}``.
        The optional ``"map"`` callable is applied to each raw line.
    """

    def __init__(self,
                 config: dict,
    ):
        super().__init__()
        self.config = config

        self.indexers = dict()
        for _, cfg in self.config.items():
            file_path = cfg['file']
            self.indexers[file_path] = FileLineIndexer(file_path)

        nl_file_name = next(iter(self.config.values()))["file"]
        self.num_lines = self.indexers[nl_file_name].line_count

    def __len__(self):
        return self.num_lines

    def __getitem__(self, idx):
        result = dict()
        for k, cfg in self.config.items():
            result[k] = self.get_sample(cfg['file'], idx)
            if "map" in cfg:
                result[k] = cfg["map"](result[k])

        return result

    def get_sample(self, file_path, idx):
        try:
            return self._get_line(file_path, idx + 1).rstrip()
        except AttributeError as e:
            print(f"Tried access line {idx + 1}")
            raise e

    def _get_line(self, file_path, idx):
        indexer = self.indexers[file_path]
        return indexer.get_line(idx)


class SourceTargetFromFiles(DataProvider):
    """DataProvider that discovers ``*.src`` / ``*.trg`` file pairs in a directory.

    File names are parsed with the pattern ``{split}_{index}_{group}.{src|trg}``.
    Each pair is wrapped in a :class:`ZipLines` dataset and registered under
    a split key like ``"train"``, ``"validation_copy"``, etc.

    :param path: Directory containing ``.src`` and ``.trg`` files.
    :param src_map: Optional callable applied to each raw source line.
    :param trg_map: Optional callable applied to each raw target line.
    :param file_subset: Per-split allowlist of file stems. If None, all
        files in the directory are used.
    """
    VALIDATION_SYN = {'validation', 'val', 'valid', 'dev'}
    TEST_SYN = {'test'}

    def __init__(
            self,
            path,
            src_map=None,
            trg_map=None,
            file_subset: dict = None
    ):
        super().__init__()
        path = os.path.join(path, '')
        src_map = (lambda x: x) if src_map is None else src_map
        trg_map = (lambda x: x) if trg_map is None else trg_map
        data_dict = {'train': [], 'validation': [], 'test': []}
        file_subset = dict() if file_subset is None else file_subset

        for file_index, file_name in enumerate(os.listdir(path)):
            show_name = ''
            info = re.findall(
                r"(?P<split>[^_.]+)_?(?P<index>\d+_)?(?P<group>[^.]*)[.](?P<srctrg>[\w\W]+)",
                file_name
            )
            file_name = re.sub("[.].+", "", file_name)
            split, index, group, srctrg  = info[0]

            if srctrg == 'trg':
                file_index = int(index[:-1]) if index else file_index

                if split in self.VALIDATION_SYN:
                    split = 'validation'
                elif split in self.TEST_SYN:
                    split = 'test'

                show_name += split

                if group:
                    show_name = show_name + '_' + group

                if split in file_subset and file_name not in file_subset[split]:
                    continue

                data_dict[split] += [(file_index, file_name, show_name)]

        # Sorting
        all_splits = []
        for _, info in data_dict.items():
            all_splits += list(sorted(info))

        self.data_dict = dict()

        for _, file_name, show_name in all_splits:
            self.data_dict[show_name] = ZipLines({
                "src": {"file": f"{path}{file_name}.src", "name": show_name, "map": src_map},
                "trg": {"file": f"{path}{file_name}.trg", "name": show_name, "map": trg_map}
            })


class TokenClassificationFromFiles(SourceTargetFromFiles):
    """SourceTargetFromFiles variant where target lines are space-separated integers."""

    def __init__(self, path: str):
        trg_map = lambda x: list(map(int, x.split()))
        super().__init__(path, trg_map=trg_map)



