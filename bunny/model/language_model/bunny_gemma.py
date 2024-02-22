import torch
from typing import Optional, Union, Tuple, List
from transformers import (
	GemmaConfig, GemmaForCausalLM, GemmaModel, AutoConfig, AutoModelForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast

from bunny.model.bunny_arch import BunnyMetaModel, BunnyMetaForCausalLM


class BunnyQwenConfig(AutoConfig):
	model_type = "bunny-gemma"


class BunnyQwenModel(BunnyMetaModel, GemmaModel):
	config_class = BunnyGemmaConfig

	def __init__(self, config: AutoConfig):
		super(BunnyQwenModel, self).__init__(config)


class BunnyGemmaForCausalLM(GemmaForCausalLM, BunnyMetaForCausalLM):
	config_class = BunnyGemmaConfig

	def __init__(self):
			pass

	def __init__(self, config):
		super(BunnyGemmaForCausalLM, self).__init__(config)
		self.model = BunnyGemmaModel(config)
		self.vocab_size = config.vocab_size
		self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size,
									bias=False)

		# Initialize weights and apply final processing
		self.post_init()

	def get_model(self):
		return self.model

	def forward(
			self,
			input_ids: torch.LongTensor = None,
			attention_mask: Optional[torch.Tensor] = None,
			position_ids: Optional[torch.LongTensor] = None,
			past_key_values: Optional[List[torch.FloatTensor]] = None,
			inputs_embeds: Optional[torch.FloatTensor] = None,
			labels: Optional[torch.LongTensor] = None,
			use_cache: Optional[bool] = None,
			output_attentions: Optional[bool] = None,
			output_hidden_states: Optional[bool] = None,
			images: Optional[torch.FloatTensor] = None,
			return_dict: Optional[bool] = None,
	) -> Union[Tuple, CausalLMOutputWithPast]:
		if inputs_embeds is None:
			print("Input embeds is none")
			(
				input_ids,
				position_ids,
				attention_mask,
				past_key_values,
				inputs_embeds,
				labels
			) = self.prepare_inputs_labels_for_multimodal(
				input_ids,
				position_ids,
				attention_mask,
				past_key_values,
				labels,
				images
			)
		return super().forward(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			labels=labels,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict
		)

	def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
									  inputs_embeds=None, attention_mask=None,
									  **kwargs):
		images = kwargs.pop("images", None)

		_inputs = super().prepare_inputs_for_generation(
			input_ids, past_key_values=past_key_values,
			inputs_embeds=inputs_embeds, attention_mask=attention_mask,
			**kwargs
		)

		if images is not None:
			print("Images is not none")
			_inputs['images'] = images
		else:
			print("Images is none")
		return _inputs


AutoConfig.register("gemma-qwen", BunnyQwenConfig)
AutoModelForCausalLM.register(BunnyGemmaConfig, BunnyGemmaForCausalLM)