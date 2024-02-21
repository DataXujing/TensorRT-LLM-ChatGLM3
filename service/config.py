#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
ChatGLM service
"""

import os


class LangChainCFG:
    work_dir = '/opt/tritonserver/chatglm3'
    
    llm_model_name = 'chatglm3-6b'
    llm_model_path = os.path.join(work_dir, llm_model_name)

    embedding_model_name = 'text2vec-large-chinese'
    embedding_model_path = os.path.join(work_dir, embedding_model_name)

    docs_path = os.path.join(work_dir, 'docs')
    knowledge_base_path = os.path.join(work_dir, 'docs')
    
    engine_dir = os.path.join(work_dir,"TensorRT-LLM-main/examples/chatglm/trt_engines/chatglm3_6b/fp16/1-gpu")