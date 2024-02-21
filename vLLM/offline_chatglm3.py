
from vllm import LLM, SamplingParams

prompts = [
    "你好",
    "你能干什么"
    
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=1024)


llm = LLM(model="/opt/tritonserver/chatglm3/chatglm3-6b",trust_remote_code=True, tokenizer_mode="auto", 
          tensor_parallel_size=1,dtype="float16")


outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")