# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import math
import time
from pathlib import Path
from typing import List

# isort: off
import torch
import torch.multiprocessing as mp
import tensorrt as trt
# isort: on
from visualize import to_onnx
from weight import get_scaling_factors, load_from_hf

try:
    import tensorrt_llm
except:
    import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import ChatGLMHeadModel, quantize_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.profiler import check_gpt_mem_usage
from tensorrt_llm.quantization import QuantMode


def get_engine_name(model, dtype, tp_size, pp_size, rank):
    if pp_size == 1:
        return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)
    return '{}_{}_tp{}_pp{}_rank{}.engine'.format(model, dtype, tp_size,
                                                  pp_size, rank)


def find_engines(dir: Path,
                 model_name: str = "*",
                 dtype: str = "*",
                 tp_size: str = "*",
                 rank: str = "*") -> List[Path]:
    template = f"{model_name}_{dtype}_tp{tp_size}_rank{rank}.engine"
    return list(dir.glob(template))


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(engine)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def truncate_input_output_len(
    max_input_len,
    max_output_len,
    max_seq_length_from_config,
    is_fixed_max_position_length=False,
):
    max_seq_length = max_seq_length_from_config
    if max_input_len >= max_seq_length_from_config:
        print("Truncate max_input_len as %d" % (max_seq_length_from_config - 1))
        max_input_len = max_seq_length_from_config - 1
        max_output_len = 1
    elif max_input_len + max_output_len > max_seq_length_from_config:
        print("Truncate max_output_len as %d" %
              (max_seq_length_from_config - max_input_len))
        max_output_len = max_seq_length_from_config - max_input_len
    elif not is_fixed_max_position_length:
        max_seq_length = max_input_len + max_output_len
    return max_input_len, max_output_len, max_seq_length


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        '-m',
        type=str,
        required=True,
        choices=[
            "chatglm_6b", "chatglm2_6b", "chatglm2_6b_32k", "chatglm3_6b",
            "chatglm3_6b_base", "chatglm3_6b_32k", "glm_10b"
        ],
        help=
        'the name of the model, use "_" rather than "-" to connect the name parts'
    )
    parser.add_argument(
        '--world_size',
        '-ws',
        type=int,
        default=1,
        help='world size, only support tensor parallelism now',
    )
    parser.add_argument('--tp_size', '-tp', type=int, default=1)
    parser.add_argument('--pp_size', '-pp', type=int, default=1)
    parser.add_argument('--model_dir', type=str, default="/opt/tritonserver/chatglm3/chatglm3-6b")
    parser.add_argument('--quant_ckpt_path', type=str, default="awq/")
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float32', 'float16', 'bfloat16'],
    )
    parser.add_argument(
        '--logits_dtype',
        type=str,
        default='float32',
        choices=['float16', 'float32'],
    )
    parser.add_argument(
        '--timing_cache',
        type=str,
        default='model.cache',
        help=
        'The path of to read timing cache from, will be ignored if the file does not exist'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='info',
        choices=['verbose', 'info', 'warning', 'error', 'internal_error'],
    )
    parser.add_argument('--max_batch_size', type=int, default=4)  # 改成4
    parser.add_argument('--max_input_len', type=int, default=1024)
    parser.add_argument('--max_output_len', type=int, default=1024)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument(
        '--use_gpt_attention_plugin',
        nargs='?',
        const='float16',
        default='float16',
        choices=['float32', 'float16', 'bfloat16', False],
        help=
        "Activates attention plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_gemm_plugin',
        nargs='?',
        const='float16',
        type=str,
        default='float16',
        choices=['float32', 'float16', 'bfloat16', False],
        help=
        "Activates GEMM plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_layernorm_plugin',
        nargs='?',
        const='float16',
        type=str,
        default='float16',
        choices=['float32', 'float16', 'bfloat16', False],
        help=
        "Activates layernorm plugin for ChatGLM-6B / GLM-10B models. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_rmsnorm_plugin',
        nargs='?',
        const='float16',
        type=str,
        default='float16',
        choices=['float32', 'float16', 'bfloat16', False],
        help=
        "Activates rmsnorm plugin for ChatGLM2-6B* / ChatGLM3-6B* models. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument('--gather_all_token_logits',
                        action='store_true',
                        default=False)
    parser.add_argument('--parallel_build', default=False, action='store_true')
    parser.add_argument(
        '--enable_context_fmha',
        default=False,
        action='store_true',
    )
    parser.add_argument(
        '--enable_context_fmha_fp32_acc',
        default=False,
        action='store_true',
    )
    parser.add_argument(
        '--multi_block_mode',
        default=False,
        action='store_true',
        help=
        'Split long kv sequence into multiple blocks (applied to generation MHA kernels). \
                        It is beneifical when batchxnum_heads cannot fully utilize GPU.'
    )
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument(
        '--enable_debug_output',
        default=False,
        action='store_true',
    )
    parser.add_argument('--gpus_per_node', type=int, default=8)
    parser.add_argument('--builder_opt', type=int, default=None)
    parser.add_argument(
        '--output_dir',
        type=Path,
        default='engine_outputs',
        help=
        'The path to save the serialized engine files, timing cache file and model configs'
    )
    parser.add_argument(
        '--strongly_typed',
        default=False,
        action="store_true",
        help=
        'This option is introduced with trt 9.1.0.1+ and will reduce the building time significantly for fp8.'
    )
    parser.add_argument(
        '--remove_input_padding',
        default=False,
        action='store_true',
    )
    parser.add_argument(
        '--paged_kv_cache',
        action="store_true",
        default=False,
        help=
        'By default we use contiguous KV cache. By setting this flag you enable paged KV cache'
    )
    parser.add_argument(
        '--use_inflight_batching',
        action="store_true",
        default=False,
        help="Activates inflight batching mode of gptAttentionPlugin.",
    )

    # Arguments related to the quantization of the model.
    parser.add_argument(
        '--use_smooth_quant',
        default=False,
        action="store_true",
        help=
        'Use the SmoothQuant method to quantize activations and weights for the various GEMMs.'
        'See --per_channel and --per_token for finer-grained quantization options.'
    )
    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision',
    )
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4', 'int4_awq'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.',
    )
    parser.add_argument(
        '--per_channel',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor for the GEMM\'s result. '
        'per_channel instead uses a different static scaling factor for each channel. '
        'The latter is usually more accurate, but a little slower.',
    )
    parser.add_argument(
        '--per_token',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor to scale activations in the int8 range. '
        'per_token chooses at run time, and for each token, a custom scaling factor. '
        'The latter is usually more accurate, but a little slower.',
    )
    parser.add_argument(
        '--per_group',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor to scale weights in the int4 range. '
        'per_group chooses at run time, and for each group, a custom scaling factor. '
        'The flag is built for GPTQ/AWQ quantization.',
    )
    parser.add_argument(
        '--group_size',
        type=int,
        default=128,
        help='Group size used in GPTQ/AWQ quantization.',
    )
    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=None,
        help=
        'Seed to use when initializing the random number generator for torch.',
    )
    parser.add_argument(
        '--tokens_per_block',
        type=int,
        default=128,
        help='Number of tokens per block in paged KV cache',
    )

    parser.add_argument(
        '--enable_fp8',
        default=False,
        action='store_true',
        help='Use FP8 Linear layer for Attention QKV/Dense and MLP.',
    )
    parser.add_argument(
        '--fp8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. fp8_kv_cache chooses fp8 quantization for KV'
    )
    parser.add_argument(
        '--max_num_tokens',
        type=int,
        default=None,
        help='Define the max number of tokens supported by the engine',
    )
    parser.add_argument(
        '--use_custom_all_reduce',
        action='store_true',
        help=
        'Activates latency-optimized algorithm for all-reduce instead of NCCL.',
    )
    args = parser.parse_args(args)

    logger.set_level(args.log_level)

    plugins_args = [
        'use_gpt_attention_plugin',
        'use_gemm_plugin',
        'use_layernorm_plugin',
        'use_rmsnorm_plugin',
    ]
    for plugin_arg in plugins_args:
        if getattr(args, plugin_arg) is None:
            logger.info(
                f"{plugin_arg} set, without specifying a value. Using {args.dtype} automatically."
            )
            setattr(args, plugin_arg, args.dtype)

    assert args.world_size == args.tp_size * args.pp_size  # only TP is supported now

    if args.model_dir is None:
        args.model_dir = args.model_name
    with open(Path(args.model_dir) / "config.json", "r") as f:
        js = json.loads(f.read())

    if args.model_name in ["chatglm_6b", "glm_10b"]:
        assert args.max_input_len < js["max_sequence_length"]

    if args.output_dir is None:
        args.output_dir = Path("output_" + args.model_name)

    if args.model_name in ["chatglm_6b"]:
        args.apply_query_key_layer_scaling = False
        args.apply_residual_connection_post_layernorm = False
        args.ffn_hidden_size = js["inner_hidden_size"]
        args.hidden_act = 'gelu'
        args.hidden_size = js["hidden_size"]
        args.linear_bias = True
        args.max_input_len, args.max_output_len, args.max_seq_length = truncate_input_output_len(
            args.max_input_len,
            args.max_output_len,
            js["max_sequence_length"],
        )
        args.multi_block_mode = False
        args.multi_query_mode = False
        args.norm_epsilon = js["layernorm_epsilon"]
        args.num_heads = js["num_attention_heads"]
        args.num_kv_heads = js["num_attention_heads"]
        args.num_layers = js["num_layers"]
        args.qkv_bias = True
        args.rmsnorm = False
        args.rotary_embedding_scaling = 1.0
        args.use_cache = js["use_cache"]
        args.vocab_size = js["vocab_size"]
    elif args.model_name in [
            "chatglm2_6b",
            "chatglm2_6b_32k",
            "chatglm3_6b",
            "chatglm3_6b_base",
            "chatglm3_6b_32k",
    ]:
        args.apply_query_key_layer_scaling = False
        args.apply_residual_connection_post_layernorm = js[
            "apply_residual_connection_post_layernorm"]
        args.ffn_hidden_size = js["ffn_hidden_size"]
        args.hidden_act = 'swiglu'
        args.hidden_size = js["hidden_size"]
        args.linear_bias = js["add_bias_linear"]
        args.max_input_len, args.max_output_len, args.max_seq_length = truncate_input_output_len(
            args.max_input_len,
            args.max_output_len,
            js["seq_length"],
        )
        args.multi_block_mode = False
        args.multi_query_mode = False  # regardless of config.json
        args.norm_epsilon = js["layernorm_epsilon"]
        args.num_heads = js["num_attention_heads"]
        args.num_kv_heads = js["multi_query_group_num"]
        args.num_layers = js["num_layers"]
        args.qkv_bias = js["add_qkv_bias"]
        args.rmsnorm = js["rmsnorm"]
        if args.model_name in ["chatglm2_6b_32k", "chatglm3_6b_32k"]:
            args.rotary_embedding_scaling = js["rope_ratio"]
        else:
            args.rotary_embedding_scaling = 1.0
        args.use_cache = js["use_cache"]
        args.vocab_size = js["padded_vocab_size"]
    elif args.model_name in ["glm_10b"]:
        args.apply_query_key_layer_scaling = False
        args.apply_residual_connection_post_layernorm = False
        args.ffn_hidden_size = 4 * js["hidden_size"]
        args.hidden_act = 'gelu'
        args.hidden_size = js["hidden_size"]
        args.linear_bias = True
        args.max_input_len, args.max_output_len, args.max_seq_length = truncate_input_output_len(
            args.max_input_len,
            args.max_output_len,
            js["max_sequence_length"],
            True,
        )
        args.multi_block_mode = False
        args.multi_query_mode = False
        args.norm_epsilon = 1.0e-5
        args.num_heads = js["num_attention_heads"]
        args.num_kv_heads = js["num_attention_heads"]
        args.num_layers = js["num_layers"]
        args.qkv_bias = True
        args.rmsnorm = False
        args.rotary_embedding_scaling = 1.0
        args.use_cache = True
        args.vocab_size = js["vocab_size"]

    if args.use_inflight_batching:
        if not args.use_gpt_attention_plugin:
            args.use_gpt_attention_plugin = 'float16'
            logger.info(
                f"Using GPT attention plugin for inflight batching mode. Setting to default '{args.use_gpt_attention_plugin}'"
            )
        if not args.remove_input_padding:
            args.remove_input_padding = True
            logger.info(
                "Using remove input padding for inflight batching mode.")
        if not args.paged_kv_cache:
            args.paged_kv_cache = True
            logger.info("Using paged KV cache for inflight batching mode.")

    assert not (
        args.use_smooth_quant and args.use_weight_only
    ), "You cannot enable both SmoothQuant and INT8 weight-only together."

    if args.use_smooth_quant:
        args.quant_mode = QuantMode.use_smooth_quant(args.per_token,
                                                     args.per_channel)
    elif args.use_weight_only:
        args.quant_mode = QuantMode.use_weight_only(
            args.weight_only_precision == 'int4')
    else:
        args.quant_mode = QuantMode(0)

    if args.int8_kv_cache:
        args.quant_mode = args.quant_mode.set_int8_kv_cache()

    elif args.fp8_kv_cache:
        args.quant_mode = args.quant_mode.set_fp8_kv_cache()
    if args.enable_fp8:
        args.quant_mode = args.quant_mode.set_fp8_qdq()

    if args.max_num_tokens is not None:
        assert args.enable_context_fmha

    assert (math.log2(args.tokens_per_block).is_integer()
            ), "tokens_per_block must be power of 2"
    if args.enable_context_fmha or args.enable_context_fmha_fp32_acc:
        assert (args.tokens_per_block >=
                128), "Context fMHA requires >= 128 tokens per block"

    logger.info(' Build Arguments '.center(100, '='))
    for k, v in vars(args).items():
        logger.info(f' - {k.ljust(30, ".")}: {v}')
    logger.info('=' * 100)

    return args


def build_rank_engine(
    builder: Builder,
    builder_config: tensorrt_llm.builder.BuilderConfig,
    engine_name: str,
    rank: int,
    args: argparse.Namespace,
) -> trt.IHostMemory:
    '''
       @brief: Build the engine on the given rank.
       @param rank: The rank to build the engine.
       @param args: The cmd line arguments.
       @return: The built engine.
    '''
    # Initialize Module
    args.mapping = Mapping(
        world_size=args.world_size,
        rank=rank,
        tp_size=args.tp_size,
    )
    assert args.num_layers % args.pp_size == 0, \
        f"num_layers {args.n_layer} must be a multiple of pipeline "\
        f"parallelism size {args.pp_size}"
    trtllm_model = ChatGLMHeadModel(
        apply_query_key_layer_scaling=args.apply_query_key_layer_scaling,
        apply_residual_connection_post_layernorm=args.
        apply_residual_connection_post_layernorm,
        dtype=args.dtype,
        enable_debug_output=args.enable_debug_output,
        ffn_hidden_size=args.ffn_hidden_size,
        hidden_act=args.hidden_act,
        hidden_size=args.hidden_size,
        linear_bias=args.linear_bias,
        logits_dtype=args.logits_dtype,
        mapping=args.mapping,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        max_seq_length=args.max_seq_length,
        model_name=args.model_name,
        norm_epsilon=args.norm_epsilon,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_layers=args.num_layers,
        qkv_bias=args.qkv_bias,
        quant_mode=args.quant_mode,
        rmsnorm=args.rmsnorm,
        rotary_embedding_scaling=args.rotary_embedding_scaling,
        tokens_per_block=args.tokens_per_block,
        use_cache=args.use_cache,
        vocab_size=args.vocab_size,
    )

    if args.use_smooth_quant or args.use_weight_only:
        trtllm_model = quantize_model(trtllm_model, args.quant_mode)
    elif args.enable_fp8 or args.fp8_kv_cache:
        logger.info(f'Loading scaling factors from '
                    f'{args.quantized_fp8_model_path}')
        quant_scales = get_scaling_factors(args.quantized_fp8_model_path,
                                           num_layers=args.n_layer,
                                           quant_mode=args.quant_mode)
        trtllm_model = quantize_model(trtllm_model,
                                      quant_mode=args.quant_mode,
                                      quant_scales=quant_scales)

    trtllm_model = load_from_hf(
        trtllm_model,
        args.model_dir,
        mapping=args.mapping,
        dtype=args.dtype,
        model_name=args.model_name,
    )

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=args.use_gpt_attention_plugin)
    if args.use_gemm_plugin:
        if not args.enable_fp8:
            network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
        else:
            logger.info(
                "Gemm plugin does not support FP8. Disabled Gemm plugin.")
    if args.use_rmsnorm_plugin:
        network.plugin_config.set_rmsnorm_plugin(dtype=args.use_rmsnorm_plugin)

    # Quantization plugins.
    if args.use_smooth_quant:
        network.plugin_config.set_smooth_quant_gemm_plugin(dtype=args.dtype)
        network.plugin_config.set_rmsnorm_quantization_plugin(dtype=args.dtype)
        network.plugin_config.set_quantize_tensor_plugin()
        network.plugin_config.set_quantize_per_token_plugin()
    assert not (args.enable_context_fmha and args.enable_context_fmha_fp32_acc)
    if args.enable_context_fmha:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    if args.enable_context_fmha_fp32_acc:
        network.plugin_config.set_context_fmha(
            ContextFMHAType.enabled_with_fp32_acc)
    if args.multi_block_mode:
        network.plugin_config.enable_mmha_multi_block_mode()
    if args.use_weight_only:
        if args.per_group:
            network.plugin_config.set_weight_only_groupwise_quant_matmul_plugin(
                dtype='float16')
        else:
            network.plugin_config.set_weight_only_quant_matmul_plugin(
                dtype='float16')
    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype,
                                              args.use_custom_all_reduce)
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()
    if args.paged_kv_cache:
        network.plugin_config.enable_paged_kv_cache(args.tokens_per_block)

    with net_guard(network):
        # Prepare
        network.set_named_parameters(trtllm_model.named_parameters())

        # Forward
        inputs = trtllm_model.prepare_inputs(
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_new_tokens=args.max_output_len,
            use_cache=True,
            max_beam_width=args.max_beam_width,
        )
        trtllm_model(*inputs)
        if args.enable_debug_output:
            # mark intermediate nodes' outputs
            for k, v in trtllm_model.named_network_outputs():
                v = v.trt_tensor
                v.name = k
                network.trt_network.mark_output(v)
                v.dtype = str_dtype_to_trt(args.dtype)
        if args.visualize:
            model_path = args.output_dir / 'test.onnx'
            to_onnx(network.trt_network, model_path)

    tensorrt_llm.graph_rewriting.optimize(network)

    # Network -> Engine
    engine = None
    engine = builder.build_engine(network, builder_config)
    if rank == 0:
        config_path = args.output_dir / 'config.json'
        builder.save_config(builder_config, config_path)

    return engine


def build(rank, args):
    torch.cuda.set_device(rank % args.gpus_per_node)
    logger.set_level(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timing_cache_file = args.timing_cache
    timing_cache = timing_cache_file

    builder = Builder()

    for cur_rank in range(args.world_size):
        # skip other ranks if parallel_build is enabled
        if args.parallel_build and cur_rank != rank:
            continue
        # NOTE: when only int8 kv cache is used together with paged kv cache no int8 tensors are exposed to TRT
        int8_trt_flag = args.quant_mode.has_act_or_weight_quant() or (
            not args.paged_kv_cache and args.quant_mode.has_int8_kv_cache())
        builder_config = builder.create_builder_config(
            precision=args.dtype,
            timing_cache=timing_cache,
            tensor_parallel=args.tp_size,
            pipeline_parallel=args.pp_size,
            int8=int8_trt_flag,
            fp8=args.enable_fp8,
            strongly_typed=args.strongly_typed,
            opt_level=args.builder_opt,
            hardware_compatibility=None,
            apply_query_key_layer_scaling=args.apply_query_key_layer_scaling,
            gather_all_token_logits=args.gather_all_token_logits,
            hidden_act=args.hidden_act,
            hidden_size=args.hidden_size,
            max_batch_size=args.max_batch_size,
            max_beam_width=args.max_beam_width,
            max_input_len=args.max_input_len,
            max_num_tokens=args.max_output_len + args.max_input_len,
            max_output_len=args.max_output_len,
            max_position_embeddings=args.max_seq_length,
            multi_query_mode=args.multi_query_mode,
            name=args.model_name,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            num_layers=args.num_layers,
            paged_kv_cache=args.paged_kv_cache,
            parallel_build=args.parallel_build,
            quant_mode=args.quant_mode,
            remove_input_padding=args.remove_input_padding,
            vocab_size=args.vocab_size,
        )

        engine_name = get_engine_name(
            args.model_name,
            args.dtype,
            args.world_size,
            args.pp_size,
            cur_rank,
        )
        engine = build_rank_engine(
            builder,
            builder_config,
            engine_name,
            cur_rank,
            args,
        )
        assert engine is not None, f'Failed to build engine for rank {cur_rank}'

        local_num_kv_heads = (args.num_kv_heads + args.world_size -
                              1) // args.world_size
        kv_dtype = str_dtype_to_trt(args.dtype)
        if args.quant_mode.has_int8_kv_cache():
            kv_dtype = str_dtype_to_trt('int8')
        elif args.quant_mode.has_fp8_kv_cache():
            kv_dtype = str_dtype_to_trt('fp8')
        check_gpt_mem_usage(
            engine=engine,
            kv_dtype=kv_dtype,
            use_gpt_attention_plugin=args.use_gpt_attention_plugin,
            paged_kv_cache=args.paged_kv_cache,
            max_batch_size=args.max_batch_size,
            max_beam_width=args.max_beam_width,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            local_num_kv_heads=local_num_kv_heads,
            head_size=args.hidden_size // args.num_heads,
            num_layers=args.num_layers)

        if cur_rank == 0:
            # Use in-memory timing cache for multiple builder passes.
            if not args.parallel_build:
                timing_cache = builder_config.trt_builder_config.get_timing_cache(
                )

        serialize_engine(engine, args.output_dir / engine_name)
        del engine

    if rank == 0:
        ok = builder.save_timing_cache(builder_config, timing_cache_file)
        assert ok, "Failed to save timing cache."


def run_build(args=None):
    args = parse_arguments(args)

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    logger.set_level(args.log_level)
    tik = time.time()
    if args.parallel_build and args.world_size > 1 and \
            torch.cuda.device_count() >= args.world_size:
        logger.warning(
            f'Parallelly build TensorRT engines. Please make sure that all of the {args.world_size} GPUs are totally free.'
        )
        mp.spawn(build, nprocs=args.world_size, args=(args, ))
    else:
        args.parallel_build = False
        logger.info('Serially build TensorRT engines.')
        build(0, args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building all {args.world_size} engines: {t}')


if __name__ == '__main__':
    run_build()
