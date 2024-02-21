'''
trt-llm调用chatglm3-6b streaming or non streaming

'''

from pathlib import Path

import numpy as np
import torch
from utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   load_tokenizer, read_model_name_from_config,
                   throttle_generator)

import tensorrt_llm
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import ModelRunner


tokenizer_dir = "/opt/tritonserver/chatglm3/chatglm3-6b"
# engine_dir = "/opt/tritonserver/chatglm3/TensorRT-LLM-main/examples/chatglm/trt_engines/chatglm3_6b/fp16/1-gpu"
engine_dir = "/opt/tritonserver/chatglm3/TensorRT-LLM-main/examples/chatglm/trt_engines/chatglm3_6b/weight_only/1-gpu"

is_streaming = False

# welcome_prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"

# # prompt + history
# def build_prompt(history):
#     prompt = welcome_prompt
#     for query, response in history:
#         prompt += f"\n\n用户：{query}"
#         prompt += f"\n\nChatGLM3-6B：{response}"
#     return prompt


# 对一个batch的text构建token input
def parse_input(tokenizer,
                input_text=None,
                prompt_template=None,
                input_file=None,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []

    for curr_text in input_text:  # input_text是个batch text
        if prompt_template is not None:
            curr_text = prompt_template.format(input_text=curr_text)
        # input_ids = tokenizer.encode(curr_text,
        #                              add_special_tokens=add_special_tokens,
        #                              truncation=True,
        #                              max_length=max_input_length)
        input_ids = tokenizer.build_chat_input(curr_text, role="user")['input_ids'][0]
        batch_input_ids.append(input_ids)

    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32).unsqueeze(0) for x in batch_input_ids
    ]
    return batch_input_ids


def print_output(tokenizer,
                 output_ids,
                 input_lengths,
                 sequence_lengths,
                 output_csv=None,
                 output_npy=None,
                 context_logits=None,
                 generation_logits=None,
                 output_logits_npy=None):
    batch_size, num_beams, _ = output_ids.size()

    for batch_idx in range(batch_size):
        inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist()  # 获取output中的input
        input_text = tokenizer.decode(inputs)                                  # 解码input到文本
        # print(f'Input [Text {batch_idx}]: \"{input_text}\"')
        
        output_texts = []
        for beam in range(num_beams):
            output_begin = input_lengths[batch_idx]
            output_end = sequence_lengths[batch_idx][beam]
            outputs = output_ids[batch_idx][beam][output_begin:output_end].tolist()  # 获取beam中的output
            output_text = tokenizer.decode(outputs)   # 解码output
            # print(
            #     f'Output [Text {batch_idx} Beam {beam}]: \"{output_text}\"')
            if not all(map(lambda c:'\u4e00' <= c <= '\u9fa5',output_text)):
                print(f"{output_text}".replace("  "," "),end=" ", flush=True)
            elif output_text.isdigit():
                print(f"{output_text}".replace("  "," "),end="", flush=True)
            else:
                print(f"{output_text}".replace("  "," "),end="", flush=True)
            
            output_texts.append(output_text)

    output_ids = output_ids.reshape((-1, output_ids.size(2)))
    
    # return output_texts[0]

def main():
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level("error")

    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=tokenizer_dir,
        vocab_file=None,
        model_name="chatglm3-6B",
    )

    runner = ModelRunner.from_dir(engine_dir=engine_dir,
                                  lora_dir=None,
                                  rank=runtime_rank,
                                  debug_mode=False)

    stop_words_list = None
    bad_words_list = None

    # if args.use_prompt_template and model_name in DEFAULT_PROMPT_TEMPLATES:
    #     prompt_template = DEFAULT_PROMPT_TEMPLATES[model_name]
    # else:
    #     prompt_template = None

    while True:
        input_text = input("\033[91m\n\n用 户：\n \033[0m")
        batch_input_ids = parse_input(tokenizer=tokenizer,
                                      input_text=[input_text],
                                      prompt_template=None,
                                      input_file=None,
                                      add_special_tokens=True,
                                      max_input_length=1024,
                                      pad_id=pad_id)
        input_lengths = [x.size(1) for x in batch_input_ids]
        # print("------------",input_text)
        # print("---------",input_lengths)

        with torch.no_grad():
            outputs = runner.generate(batch_input_ids,
                                      max_new_tokens=1024,
                                      max_kv_cache_length=None,
                                      end_id=end_id,
                                      pad_id=pad_id,
                                      temperature=0.01,
                                      top_k=1,
                                      top_p=1,
                                      num_beams=1,
                                      length_penalty=1.0,
                                      repetition_penalty=1.0,
                                      stop_words_list=stop_words_list,
                                      bad_words_list=bad_words_list,
                                      lora_uids=None,
                                      prompt_table_path=None,
                                      prompt_tasks=None,
                                      streaming=is_streaming,
                                      output_sequence_lengths=True,
                                      return_dict=True)
            torch.cuda.synchronize()

        print("\033[92m\nChatGLM：\033[0m", end="")

        if runtime_rank == 0:
            if is_streaming:
                output_len = input_lengths
                for curr_outputs in throttle_generator(outputs,1):
                    output_ids = curr_outputs['output_ids']
                    sequence_lengths = curr_outputs['sequence_lengths']
                    
                    print_output(tokenizer,
                                 output_ids,
                                 output_len,#input_lengths,
                                 sequence_lengths,
                                 output_csv=None,
                                 output_npy=None)
                    output_len = [sequence_lengths.cpu().numpy()[0][0]]
                    # print("-----------",output_len)
                   
            else:
                output_ids = outputs['output_ids']
                sequence_lengths = outputs['sequence_lengths']
                context_logits = None
                generation_logits = None
                if runner.session.gather_all_token_logits:
                    context_logits = outputs['context_logits']
                    generation_logits = outputs['generation_logits']
                print_output(tokenizer,
                             output_ids,
                             input_lengths,
                             sequence_lengths,
                             output_csv=None,
                             output_npy=None,
                             context_logits=context_logits,
                             generation_logits=generation_logits,
                             output_logits_npy=None)


if __name__ == '__main__':
    main()
