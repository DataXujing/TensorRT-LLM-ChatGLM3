
from langchain_community.llms import VLLM

llm = VLLM(model="/opt/tritonserver/chatglm3/chatglm3-6b",
           trust_remote_code=True,  # mandatory for hf models
           max_new_tokens=1024,
           top_k=10,
           top_p=0.95,
           temperature=0.8,
           tokenizer_mode="auto", 
        tensor_parallel_size=1,dtype="float16"
)

print(llm("你好"))
print(llm("你能干什么"))
