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

"""Tests for text tokenizers."""

import os
import unittest

from absl.testing import parameterized
import numpy as np
import pytest
from videoprism import tokenizers

# Check if TensorFlow is available for optional tests
try:
  import tensorflow as tf
  HAS_TF = True
except ImportError:
  HAS_TF = False


class PyToTfWrapper:
  """Allows to use `to_int_tf_op()` via `to_int()`."""

  def __init__(self, model):
    self.model = model
    self.bos_token = model.bos_token
    self.eos_token = model.eos_token
    self.vocab_size = model.vocab_size

  def to_int(self, text, *, bos=False, eos=False):
    ret = self.model.to_int_tf_op(text, bos=bos, eos=eos)
    if isinstance(ret, tf.RaggedTensor):
      return [t.numpy().tolist() for t in ret]
    return ret.numpy().tolist()


class TokenizersTest(parameterized.TestCase):

  def setUp(self):
    self.spm_path = os.path.join(
        os.path.dirname(__file__), "assets", "testdata", "test_spm.model"
    )
    super().setUp()

  def test_sentencepiece_tokenizer(self):
    """Test basic tokenization without TensorFlow."""
    model = tokenizers.SentencePieceTokenizer(self.spm_path)
    self.assertEqual(model.vocab_size, 1000)
    bos, eos = model.bos_token, model.eos_token
    self.assertEqual(bos, 1)
    self.assertEqual(eos, 2)
    self.assertEqual(model.to_int("blah"), [80, 180, 60])
    self.assertEqual(model.to_int("blah", bos=True), [bos, 80, 180, 60])
    self.assertEqual(model.to_int("blah", eos=True), [80, 180, 60, eos])
    self.assertEqual(
        model.to_int("blah", bos=True, eos=True), [bos, 80, 180, 60, eos]
    )
    self.assertEqual(
        model.to_int(["blah", "blah blah"]),
        [[80, 180, 60], [80, 180, 60, 80, 180, 60]],
    )

  @unittest.skipUnless(HAS_TF, "TensorFlow not installed")
  def test_sentencepiece_tokenizer_tf_op(self):
    """Test TensorFlow ops (requires TensorFlow)."""
    model = tokenizers.SentencePieceTokenizer(self.spm_path)
    wrapped_model = PyToTfWrapper(model)
    
    self.assertEqual(wrapped_model.vocab_size, 1000)
    bos, eos = wrapped_model.bos_token, wrapped_model.eos_token
    self.assertEqual(bos, 1)
    self.assertEqual(eos, 2)
    self.assertEqual(wrapped_model.to_int("blah"), [80, 180, 60])
    self.assertEqual(wrapped_model.to_int("blah", bos=True), [bos, 80, 180, 60])
    self.assertEqual(wrapped_model.to_int("blah", eos=True), [80, 180, 60, eos])
    self.assertEqual(
        wrapped_model.to_int("blah", bos=True, eos=True), [bos, 80, 180, 60, eos]
    )
    self.assertEqual(
        wrapped_model.to_int(["blah", "blah blah"]),
        [[80, 180, 60], [80, 180, 60, 80, 180, 60]],
    )

  @unittest.skipUnless(HAS_TF, "TensorFlow not installed")
  def test_sentencepiece_tokenizer_tf_data(self):
    """Test TensorFlow data pipeline integration (requires TensorFlow)."""
    model = tokenizers.SentencePieceTokenizer(self.spm_path)

    def gen():
      yield tf.convert_to_tensor(["blah"])
      yield tf.convert_to_tensor(["blah", "blah blah"])

    ds = tf.data.Dataset.from_generator(gen, tf.string, tf.TensorShape([None]))
    ds = ds.map(model.to_int_tf_op)
    res = [
        [b.tolist() if isinstance(b, np.ndarray) else b for b in a.tolist()]
        for a in ds.as_numpy_iterator()
    ]
    print(res)
    np.testing.assert_array_equal(
        res, [[[80, 180, 60]], [[80, 180, 60], [80, 180, 60, 80, 180, 60]]]
    )


if __name__ == "__main__":
  unittest.main()
