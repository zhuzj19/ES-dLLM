import torch.nn as nn
import types
import models.llada_func as llada_func
import models.dream_func as dream_func

def transform_model_blocks(model: nn.Module, submodule_name: str, attr_name: str, new_func, block_submodule_name: str = ''):
    blocks = model.get_submodule(submodule_name)
    for block in blocks:
        if block_submodule_name:
            block = block.get_submodule(block_submodule_name)
        bound_newfunc = types.MethodType(new_func, block)
        setattr(block, attr_name, bound_newfunc)

def transform_model_submodule(model: nn.Module, submodule_name: str, attr_name: str, new_func):
    """
    Transform a specific submodule of the model.
    """
    submodule = model.get_submodule(submodule_name)
    bound_newfunc = types.MethodType(new_func, submodule)
    setattr(submodule, attr_name, bound_newfunc)

def transform_llada_model(model: nn.Module):
    """
    Transform the model to use ES-dLLM.
    """
    # Transform LLaDALlamaBlock forward method
    transform_model_blocks(model, 'model.transformer.blocks', 'forward', llada_func.LLaDALlamaBlock_forward)
    # Transform LLaDALlamaBlock attention method
    transform_model_blocks(model, 'model.transformer.blocks', 'attention', llada_func.LLaDABlock_attention)
    # Transform RotaryEmbedding forward method
    transform_model_blocks(model, 'model.transformer.blocks', 'forward', llada_func.RotaryEmbedding_forward, 'rotary_emb')
    # Transform LLaDAModel forward method
    transform_model_submodule(model, 'model', 'forward', llada_func.LLaDAModel_forward)
    transform_model_submodule(model, '', 'forward', llada_func.LLaDAModelLM_forward)

def transform_dream_model(model: nn.Module):
    """
    Transform the model to use ES-dLLM.
    """
    transform_model_submodule(model, '', 'forward', dream_func.DreamModel_forward)
    transform_model_submodule(model, 'model', 'forward', dream_func.DreamBaseModel_forward)
    transform_model_blocks(model, 'model.layers', 'forward', dream_func.DreamDecoderLayer_forward)
    transform_model_blocks(model, 'model.layers', 'forward', dream_func.DreamSdpaAttention_forward, 'self_attn')