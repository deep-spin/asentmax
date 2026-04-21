from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from ..utils.utils import import_class_by_str


class HFWrapper(nn.Module):
    def __init__(self, model_class: str, from_pretrained: str,  vocab_size: int, **model_kwargs):
        super().__init__()
        model_kwargs["vocab_size"] = vocab_size

        config = AutoConfig.from_pretrained(
            from_pretrained,
            **{name: model_kwargs[name] for name in ["attn_implementation"] if name in model_kwargs}
        )
        config.update(model_kwargs)
        # Initializing model from string
        self.model = import_class_by_str(model_class)(config)

    def forward(self, input_ids, input_mask, **kwargs):
        return self.model.forward(input_ids, attention_mask=input_mask, **kwargs)

    def generate(self, input_ids, input_mask, **kwargs):
        return self.model.generate(
                input_ids,
                attention_mask=input_mask,
                **kwargs
        )