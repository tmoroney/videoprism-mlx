# Copyright 2025 VideoPrism Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for checkpointing and other purposes."""

import collections
from collections.abc import Mapping, Sequence
import hashlib
import io
import os
import string
import tempfile
from urllib import parse as urlparse

import fsspec
import jax
import numpy as np


def traverse_with_names(tree, with_inner_nodes=False):
  """Traverses nested dicts and emits (leaf_name, leaf_val).

  Args:
    tree: JAX Pytree object.
    with_inner_nodes: Whether to traverse the non-leaf nodes.

  Yields:
    A pair of (leaf_name, leaf_val).
  """
  # Don't output the non-leaf nodes. If the optimizer doesn't have a state
  # the tree leaves can be Nones which was interpreted as a leaf by this
  # function but not by the other functions (like jax.tree.map).
  if tree is None:
    return
  elif isinstance(tree, Mapping):
    keys = sorted(tree.keys())
    for key in keys:
      for path, v in traverse_with_names(tree[key], with_inner_nodes):
        yield (key + "/" + path).rstrip("/"), v
    if with_inner_nodes:
      yield "", tree
  elif isinstance(tree, Sequence):
    for idx in range(len(tree)):
      for path, v in traverse_with_names(tree[idx], with_inner_nodes):
        yield (str(idx) + "/" + path).rstrip("/"), v
    if with_inner_nodes:
      yield "", tree
  else:
    yield "", tree


def tree_flatten_with_names(tree):
  """Populates tree_flatten with leaf names.

  Args:
    tree: JAX Pytree object.

  Returns:
    A list of values with names: [(name, value), ...]
  """
  vals, tree_def = jax.tree.flatten(tree)

  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)
  val_names, perm = zip(*traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)

  # Custom traverasal should visit the same number of leaves.
  assert len(val_names) == len(vals)

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)]


def recover_tree(keys, values):
  """Recovers a tree as a nested dict from flat names and values.

  Args:
    keys: A list of keys, where '/' is used as separator between nodes.
    values: A list of leaf values.

  Returns:
    A nested tree-like dict.
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if "/" not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split("/", 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = recover_tree(k_subtree, v_subtree)
  return tree


def _get_cache_dir() -> str:
  env_dir = os.environ.get("VIDEOPRISM_CACHE_DIR")
  if env_dir:
    return env_dir
  home_dir = os.path.expanduser("~")
  if home_dir and home_dir != "~" and os.path.isdir(home_dir):
    return os.path.join(home_dir, ".cache", "videoprism")
  return os.path.join(tempfile.gettempdir(), "videoprism_cache")


_CACHE_DIR = _get_cache_dir()
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_remote_file(path: str) -> str:
  if not path.startswith(("gs://", "http://", "https://", "s3://")):
    return path

  parsed = urlparse.urlparse(path)
  ext = os.path.splitext(parsed.path)[1] or ".cache"
  digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
  local_path = os.path.join(_CACHE_DIR, f"{digest}{ext}")

  if os.path.exists(local_path):
    return local_path

  storage_options = {"token": "anon"} if path.startswith("gs://") else {}
  with fsspec.open(path, "rb", **storage_options) as src, tempfile.NamedTemporaryFile(
      dir=_CACHE_DIR, suffix=ext, delete=False
  ) as dst:
    dst.write(src.read())
    temp_path = dst.name

  os.replace(temp_path, local_path)
  return local_path


def npload(fname):
  """Loads `fname` and returns an np.ndarray or dict thereof."""
  full_path = _cache_remote_file(fname)
  loaded = np.load(full_path, allow_pickle=False)

  # Support loading both single-array files (np.save) and zips (np.savez).
  if isinstance(loaded, np.ndarray):
    return loaded
  else:
    return dict(loaded)


def load_checkpoint(npz):
  """Loads a jax Pytree from a npz file.

  Args:
    npz: Either path to the checkpoint file (.npz), or a dict-like.

  Returns:
    A Pytree that is the checkpoint.
  """
  if isinstance(npz, str):  # If not already loaded, then load.
    npz = npload(npz)
  keys, values = zip(*list(npz.items()))
  return recover_tree(keys, values)


def canonicalize_text(text: str) -> str:
  """Canonicalizes text.

  Canonicalization includes:
  - Replace all punctuation with a whitespace.
  - Use all lower case.
  - Leave only one whitespace between words.
  - End with a period.

  Examples:
    "Hello, World!" -> "hello world."
    "Hello,World.." -> "hello world."
    "  Hello   WORLD" -> "hello world."

  Args:
    text: A string for the input text.

  Returns:
    A string for the canonicalized text.
  """
  # Replace all punctuation with a whitespace.
  p = string.punctuation
  text = text.translate(str.maketrans(p, " " * len(p)))
  # Use all lower case.
  text = text.lower()
  # Leave only one whitespace between words.
  text = " ".join(text.split())
  # End with a period.
  text = text + "."
  return text
