from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer
model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-2.8b-slimpj")
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-slimpj")
print(model)