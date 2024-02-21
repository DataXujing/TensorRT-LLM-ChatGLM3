
# 这个代码无法执行，autoAWQ暂不支持chatglm3
from awq import AutoAWQForCausalLM,AutoModel
from transformers import AutoTokenizer

model_path = "/opt/tritonserver/chatglm3/chatglm3-6b"
quant_path = "/opt/tritonserver/chatglm3/chatglm3-6b-awq"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
# model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)