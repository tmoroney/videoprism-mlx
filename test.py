import jax
from videoprism import models as vp

# Video-text encoders.
model_name = 'videoprism_lvt_public_v1_base'  # configuration name
flax_model = vp.get_model(model_name)
loaded_state = vp.load_pretrained_weights(model_name)
text_tokenizer = vp.load_text_tokenizer('c4_en')

@jax.jit
def forward_fn(inputs, text_token_ids, text_token_paddings, train=False):
  return flax_model.apply(
      loaded_state,
      inputs,
      text_token_ids,
      text_token_paddings,
      train=train,
  )

video_inputs = ...  # Shape = [batch_size, num_frames, height, width, 3].
text_queries = ...  # A list of input text queries.
text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, text_queries)
video_embeddings, text_embeddings, _ = forward_fn(video_inputs, text_ids, text_paddings)  # Shape = [batch_size, feature_channels].