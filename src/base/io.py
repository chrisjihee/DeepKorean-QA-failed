import json
import os
import shutil
import sys
from itertools import chain
from pathlib import Path
from time import sleep

from attrdict import AttrDict


class RedirStd(object):
    def __init__(self, stdout=None, stderr=None):
        self._mute = open(os.devnull, 'w')
        self._stdout = stdout or self._mute
        self._stderr = stderr or self._mute
        sleep(0.5)

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        sleep(0.5)
        self._stdout.flush()
        self._stderr.flush()
        self._mute.close()
        sys.stdout, sys.stderr = self.old_stdout, self.old_stderr


def dirs(path):
    path = Path(path)
    if any(c in str(path.parent) for c in ["*", "?"]):
        parents = dirs(path.parent)
    else:
        parents = [path.parent]
    return sorted([x for x in chain.from_iterable(parent.glob(path.name) for parent in parents) if x.is_dir()])


def files(path):
    path = Path(path)
    if any(c in str(path.parent) for c in ["*", "?"]):
        parents = dirs(path.parent)
    else:
        parents = [path.parent]
    return sorted([x for x in chain.from_iterable(parent.glob(path.name) for parent in parents) if x.is_file()])


def glob_dirs(path, glob: str):
    path = Path(path)
    return sorted([x for x in path.glob(glob) if x.is_dir()])


def glob_files(path, glob: str):
    path = Path(path)
    return sorted([x for x in path.glob(glob) if x.is_file()])


def new_path(path, post=None) -> Path:
    path = Path(path)
    return path.parent / (path.stem + (f"-{post}" if post else "") + ''.join(path.suffixes))


def new_file(infile, outfiles, blank=('', '*', '?')) -> Path:
    infile = Path(infile)
    outfiles = Path(outfiles)
    parent = outfiles.parent
    parent.mkdir(parents=True, exist_ok=True)

    suffix1 = ''.join(infile.suffixes)
    suffix2 = ''.join(outfiles.suffixes)
    suffix = suffix1 if suffix2 in blank else suffix2

    stem1 = infile.stem.strip()
    stem2 = outfiles.stem.strip()
    stem = stem1 if any(x and x in stem2 for x in blank) else stem2

    outfile: Path = parent / f"{stem}{suffix}"
    assert infile != outfile, f"infile({infile}) == outfile({outfile})"

    return outfile


def make_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_parent_dir(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def remove_dir(path, ignore_errors=False) -> Path:
    path = Path(path)
    shutil.rmtree(path, ignore_errors=ignore_errors)
    return path


def remove_any(path) -> Path:
    path = Path(path)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    return path


def load_json(path, **kwargs) -> dict:
    file = Path(path)
    assert file.exists() and file.is_file()
    with file.open() as f:
        return json.load(f, **kwargs)


def save_json(obj, path, **kwargs):
    file = make_parent_dir(Path(path))
    assert not file.exists() or file.is_file()
    with file.open("w") as f:
        json.dump(obj, f, **kwargs)


def merge_dicts(*args):
    items = list()
    for x in args:
        if x is not None:
            items += x.items()
    return dict(items)


def load_attrs(file, pre=None, post=None) -> AttrDict:
    return AttrDict(merge_dicts(pre, load_json(file), post))


def save_rows(rows, file, keys=None, excl=None, with_column_name=False):
    first = next(rows)
    if keys is not None and isinstance(keys, (list, tuple, set)):
        keys = [x for x in keys if x in first.keys()]
    else:
        keys = first.keys()
    if excl is not None and isinstance(excl, (list, tuple, set)):
        keys = [x for x in keys if x not in excl]
    with file.open("a") as out:
        if with_column_name:
            print('\t'.join(keys), file=out)
        for row in chain([first], rows):
            print('\t'.join(map(str, [row[k] for k in keys])), file=out)


def save_attrs(obj: AttrDict, file, keys=None, excl=None):
    if keys is not None and isinstance(keys, (list, tuple, set)):
        keys = [x for x in keys if x in obj.keys()]
    else:
        keys = obj.keys()
    if excl is not None and isinstance(excl, (list, tuple, set)):
        keys = [x for x in keys if x not in excl]
    save_json({key: obj[key] for key in keys}, file, ensure_ascii=False, indent=4, default=str)


def set_cuda_path(candidates=("/usr/local/cuda-11.4", "/usr/local/cuda-11.3", "/usr/local/cuda-11.1", "/usr/local/cuda-10.2")):
    cuda_dir = None
    for candidate in candidates:
        if Path(candidate).exists():
            cuda_dir = candidate
            break
    assert cuda_dir is not None
    os.environ['PATH'] = f"{cuda_dir}/bin:{os.environ['PATH']}"


def set_torch_ext_path(n_run=1):
    torch_ext_dir = Path(f"cache/torch_extensions/n_run={n_run}")
    os.environ['TORCH_EXTENSIONS_DIR'] = f"{torch_ext_dir.absolute()}"
