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

"""Tests for encoder modules."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import numpy as jnp
import numpy as np
from videoprism import encoders


class EncodersTest(parameterized.TestCase):

  @chex.variants(with_jit=True, without_jit=True)
  def test_trainable_positional_embedding_layer(self):
    seq_len, dim = 8, 10
    prng_key = jax.random.PRNGKey(seed=123)
    emb_layer = encoders.TrainablePositionalEmbedding(
        name='pos_emb',
        max_seq_length=seq_len,
        embedding_dim=dim,
        lookup_style='matmul',
    )

    @self.variant
    def var_fn():
      return emb_layer.init_with_output(prng_key, seq_len)

    outputs, params = var_fn()
    self.assertLen(jax.tree_util.tree_flatten(params)[0], 1)
    self.assertEqual(outputs.shape, (1, seq_len, dim))

  @chex.variants(with_jit=True)
  @parameterized.product(
      scan=[True, False],
      train=[True, False],
  )
  def test_vision_transformer(self, scan: bool, train: bool):
    batch_size, seq_len, dim = 1, 6, 4
    np_inputs = np.random.normal(1.0, 0.5, [batch_size, seq_len, dim]).astype(
        'float32'
    )
    inputs = jnp.asarray(np_inputs)
    prng_key = jax.random.PRNGKey(seed=123)
    vit = encoders.VisionTransformer(
        name='vit',
        num_tfm_layers=2,
        mlp_dim=4,
        num_heads=2,
        scan=scan,
    )

    @self.variant
    def var_fn():
      return vit.init_with_output(prng_key, inputs, train=train)

    outputs, params = var_fn()
    self.assertLen(jax.tree_util.tree_flatten(params)[0], 16 if scan else 32)
    self.assertEqual(outputs.shape, (batch_size, seq_len, dim))

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      ('train', False, True, False, False),
      ('scan', True, False, False, False),
      ('scan_and_train', True, True, False, False),
      ('return_intermediate', True, False, True, False),
      ('use_frame_paddings', True, False, False, True),
  )
  def test_factorized_encoder(
      self,
      scan: bool,
      train: bool,
      return_intermediate: bool,
      use_frame_paddings: bool,
  ):
    batch_size, num_frames, image_size, patch_size, dim = 1, 4, 16, 4, 8
    np_inputs = np.random.normal(
        0.0,
        0.1,
        [batch_size, num_frames, image_size, image_size, 3],
    ).astype('float32')
    inputs = jnp.asarray(np_inputs)

    frame_paddings = None
    if use_frame_paddings:
      np_frame_paddings = np.zeros((batch_size, num_frames), dtype='float32')
      np_frame_paddings[:, num_frames // 2 :] = 1
      frame_paddings = jnp.asarray(np_frame_paddings)

    prng_key = jax.random.PRNGKey(seed=123)
    enc = encoders.FactorizedEncoder(
        name='enc',
        patch_size=patch_size,
        pos_emb_shape=(16, 16, 16),
        model_dim=dim,
        num_spatial_layers=2,
        num_temporal_layers=2,
        num_heads=2,
        mlp_dim=4,
        atten_logit_cap=50.0,
        scan=scan,
    )

    @self.variant
    def var_fn():
      return enc.init_with_output(
          prng_key,
          inputs,
          train=train,
          return_intermediate=return_intermediate,
          frame_paddings=frame_paddings,
      )

    if return_intermediate:
      (outputs, intermediate_outputs), params = var_fn()
    else:
      outputs, params = var_fn()
      intermediate_outputs = None

    self.assertLen(jax.tree_util.tree_flatten(params)[0], 40 if scan else 72)
    self.assertEqual(
        outputs.shape,
        (batch_size, num_frames * (image_size // patch_size) ** 2, dim),
    )
    if return_intermediate:
      spatial_features = intermediate_outputs['spatial_features']
      self.assertEqual(
          spatial_features.shape,
          (batch_size, num_frames * (image_size // patch_size) ** 2, dim),
      )


if __name__ == '__main__':
  absltest.main()
