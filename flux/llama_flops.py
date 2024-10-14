import torch
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
from xfuser.model_executor.pipelines.flops_profiler import get_model_profile

tokenizer = LlamaTokenizer.from_pretrained("/home/nvme-share/share/models/Llama-2-7b-hf")
model = LlamaForCausalLM(LlamaConfig())
model.bos_token_id = tokenizer.bos_token_id
model.eos_token_id = tokenizer.eos_token_id
model.pad_token_id = tokenizer.eos_token_id
model.cuda()

length = 1024
input_ids = torch.randint(0, 100, (8, length)).cuda()


# warm up
# for _ in range(10):
#     model.generate(input_ids, max_length=length + 1)

# # profile
# get_model_profile(model.generate, input_ids, kwargs={"max_length": 1}, mode='generate')

