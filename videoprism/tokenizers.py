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

"""Tokenizers for text encoders."""

from collections.abc import Sequence
from typing import Protocol, TYPE_CHECKING

import sentencepiece
from huggingface_hub import hf_hub_download

if TYPE_CHECKING:
  import tensorflow as tf

SentencePieceProcessor = sentencepiece.SentencePieceProcessor


class Tokenizer(Protocol):
  """Tokenizer interface."""

  def to_int(
      self, text: str | Sequence[str], *, bos: bool = False, eos: bool = False
  ) -> list[int] | list[list[int]]:
    """Tokenizes `text` into a list of integer tokens.

    Args:
      text: can be a single string, or a list of strings.
      bos: Whether a beginning-of-sentence token should be prepended.
      eos: Whether an end-of-sentence token should be appended.

    Returns:
      A list or list-of-list of tokens.
    """

  def to_int_tf_op(
      self, text: str | Sequence[str], *, bos: bool = False, eos: bool = False
  ) -> "tf.Tensor | tf.RaggedTensor":
    """Same as `to_int()`, but as TF ops to be used in data pipelines.

    Note: This method requires TensorFlow to be installed. It is optional
    and only needed if you're using TensorFlow data pipelines.

    Args:
      text: can be a single string, or a list of strings.
      bos: Whether a beginning-of-sentence token should be prepended.
      eos: Whether an end-of-sentence token should be appended.

    Returns:
      A tf.Tensor of tokens.
    """

  @property
  def pad_token(self) -> int:
    """Token id of padding token."""

  @property
  def eos_token(self) -> int:
    """Token id of end-of-sentence token."""

  @property
  def bos_token(self) -> int:
    """Token id of beginning-of-sentence token."""

  @property
  def vocab_size(self) -> int:
    """Returns the size of the vocabulary."""


class SentencePieceTokenizer(Tokenizer):
  """Wraps a SentencePiece model for tokenization."""

  def __init__(self, model_path="c4_en.model"):
    """Initializes the tokenizer.

    Args:
      model_path: Filename of the SentencePiece model to download from HuggingFace.
                  If a GCS path is provided (gs://...), it will be ignored and the
                  default c4_en.model will be used instead.
    """
    # Handle legacy GCS paths by using the default model
    if model_path.startswith("gs://"):
      model_path = "c4_en.model"
    
    local_model_path = hf_hub_download(
        repo_id="tom-moroney/videoprism-mlx",
        filename=model_path
    )
    self._model = SentencePieceProcessor()
    self._model.Load(local_model_path)

  def to_int(
      self, text: str | Sequence[str], *, bos: bool = False, eos: bool = False
  ) -> list[int] | list[list[int]]:
    """Tokenizes `text` into a list of integer tokens.

    Args:
      text: can be a single string, or a list of strings.
      bos: Whether a beginning-of-sentence token should be prepended.
      eos: Whether an end-of-sentence token should be appended.

    Returns:
      A list or list-of-list of tokens.
    """

    def _single(s: str) -> list[int]:
      return (
          ([self.bos_token] if bos else [])
          + self._model.EncodeAsIds(s)
          + ([self.eos_token] if eos else [])
      )

    if isinstance(text, str):
      return _single(text)
    return list([_single(s) for s in text])

  def to_int_tf_op(
      self, text: str | Sequence[str], *, bos: bool = False, eos: bool = False
  ) -> "tf.Tensor | tf.RaggedTensor":
    """Same as `to_int()`, but as TF ops to be used in data pipelines.

    Note: This method requires TensorFlow to be installed. It is optional
    and only needed if you're using TensorFlow data pipelines.

    Args:
      text: can be a single string, or a list of strings.
      bos: Whether a beginning-of-sentence token should be prepended.
      eos: Whether an end-of-sentence token should be appended.

    Returns:
      A tf.Tensor or tf.RaggedTensor of tokens.
    
    Raises:
      ImportError: If TensorFlow is not installed.
    """
    try:
      import tensorflow as tf
    except ImportError as e:
      raise ImportError(
          "TensorFlow is required for to_int_tf_op(). "
          "Install it with: pip install tensorflow or tensorflow-cpu"
      ) from e
    
    text = tf.convert_to_tensor(text)
    if text.ndim == 0:

      def fn(txt):
        """Tokenizes a single string."""
        s = txt.numpy().decode()
        return tf.constant(self.to_int(s, bos=bos, eos=eos), tf.int32)

      return tf.py_function(fn, [text], tf.int32)
    else:

      def fn(txt):
        """Tokenizes a list of strings."""
        strings = [s.decode() for s in txt.numpy().tolist()]
        toks = self.to_int(strings, bos=bos, eos=eos)
        return tf.ragged.constant(toks)

      out_type = tf.RaggedTensorSpec([text.shape[0], None], tf.int32)
      return tf.py_function(fn, [text], Tout=out_type)

  @property
  def pad_token(self) -> int:
    """Token id of padding token."""
    return self._model.pad_id()

  @property
  def eos_token(self) -> int:
    """Token id of end-of-sentence token."""
    return self._model.eos_id()

  @property
  def bos_token(self) -> int:
    """Token id of beginning-of-sentence token."""
    return self._model.bos_id()

  @property
  def vocab_size(self) -> int:
    """Returns the size of the vocabulary."""
    return self._model.GetPieceSize()
