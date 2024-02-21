# hf 查看模型结构

import os
import platform
from transformers import AutoTokenizer, AutoModel
MODEL_PATH = "/opt/tritonserver/chatglm3/chatglm3-6b"
TOKENIZER_PATH = "/opt/tritonserver/chatglm3/chatglm3-6b"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

print(model)
