#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
ChatGLM service 基于TensorRT LLM

我可以换成一个云服务的LLM
"""

from typing import Dict, Union, Optional
from typing import List

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoModel, AutoTokenizer

import torch
from service.utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   load_tokenizer, read_model_name_from_config,
                   throttle_generator)

import tensorrt_llm
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import ModelRunner


class ChatGLMService(LLM):
    max_token: int = 1024
    temperature: float = 0.1
    top_p = 0.9
    history = []
    tokenizer: object = None
    model: object = None
    runner: object = None
    end_id = 0
    pad_id = 0
    
    # stop_words_list = None
    # bad_words_list = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        
        batch_input_ids = parse_input(tokenizer=self.tokenizer,
                                      input_text=[prompt],
                                      prompt_template=None,
                                      input_file=None,
                                      add_special_tokens=True,
                                      max_input_length=1024,
                                      pad_id=self.pad_id)
        input_lengths = [x.size(1) for x in batch_input_ids]
        
        
        with torch.no_grad():
            outputs = self.runner.generate(batch_input_ids,
                                      max_new_tokens=self.max_token,
                                      max_kv_cache_length=None,
                                      end_id=self.end_id,
                                      pad_id=self.pad_id,
                                      temperature=self.temperature,
                                      top_k=1,
                                      top_p=self.top_p,
                                      num_beams=1,
                                      length_penalty=1.0,
                                      repetition_penalty=1.0,
                                      stop_words_list=None,
                                      bad_words_list=None,
                                      lora_uids=None,
                                      prompt_table_path=None,
                                      prompt_tasks=None,
                                      streaming=False,
                                      output_sequence_lengths=True,
                                      return_dict=True)
            torch.cuda.synchronize()
            
        output_ids = outputs['output_ids']
        sequence_lengths = outputs['sequence_lengths']
        context_logits = None
        generation_logits = None
        if self.runner.session.gather_all_token_logits:
            context_logits = outputs['context_logits']
            generation_logits = outputs['generation_logits']
        output_text = print_output(self.tokenizer,
                        output_ids,
                        input_lengths,
                        sequence_lengths,
                        output_csv=None,
                        output_npy=None,
                        context_logits=context_logits,
                        generation_logits=generation_logits,
                        output_logits_npy=None)
        
        self.history = self.history + [(prompt,output_text)] 
        
        return output_text
        

    def load_model(self,
                   model_name_or_path: str = "chatglm3-6b",engine_dir: str="chatglm3-6b"):
        runtime_rank = tensorrt_llm.mpi_rank()
        logger.set_level("error")

        self.tokenizer, self.pad_id, self.end_id = load_tokenizer(
            tokenizer_dir=model_name_or_path,
            vocab_file=None,
            model_name="chatglm3-6B",
        )
        
        self.runner = ModelRunner.from_dir(engine_dir=engine_dir,
                                    lora_dir=None,
                                    rank=runtime_rank,
                                    debug_mode=False)


# input and output
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
            # # print(
            # #     f'Output [Text {batch_idx} Beam {beam}]: \"{output_text}\"')
            # if not all(map(lambda c:'\u4e00' <= c <= '\u9fa5',output_text)):
            #     print(f"{output_text}".replace("  "," "),end=" ", flush=True)
            # elif output_text.isdigit():
            #     print(f"{output_text}".replace("  "," "),end="", flush=True)
            # else:
            #     print(f"{output_text}".replace("  "," "),end="", flush=True)
            
            output_texts.append(output_text)

    output_ids = output_ids.reshape((-1, output_ids.size(2)))
    
    return output_texts[0]