from __future__ import annotations

import os
import regex as re
from collections.abc import Iterable, Iterator
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    bpe = BpeTokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
    return bpe
    

# class PreTokenizer:
#     def __init__(self):
#         self.pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# class BpeTokenizer:
#     def __init__(
#         self,
#         vocab: dict[int, bytes],
#         merges: list[tuple[bytes, bytes]],
#         special_tokens: list[str] | None = None,
#     ):
#         self.vocab = vocab
#         self.merges = merges
#         self.special_tokens = special_tokens or []
#         # 创建一个vocab的bytes到id的反向映射
#         self.rev_vocab = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
#     def from_files(cls, vocab_filepath: str, merges_filepath:str, special_tokens: list[str] | None = None):
#         # 暂时没有函数调用这个，所以先不实现了
#         pass
#     # def encode(self, text: str) -> list[int]: 
#     #     # 总任务: 将输入文本编码成 token id 列表
#     #     # 1. 先标记特殊字符
#     #     text_split_by_specialToken = split_text_by_special_tokens(text, self.special_tokens)

        
#     #     token_ids: list[int] = []
#     #     # 3. 对于每一个pre_token，不停地循环其能够使用的merge 策略，直到无法使用为止
#     #     for idx1, (text_str, is_special) in enumerate(text_split_by_specialToken):
#     #         if is_special:
#     #             # 特殊将special按照vocab处理成一个token id
#     #             special_token_id = self.rev_vocab.get(text_str.encode("utf-8"), -1)
#     #             if special_token_id == -1:
#     #                 raise ValueError(f"Special token {text_str} not found in vocabulary.")
#     #             token_ids.append(special_token_id)
#     #             continue
#     #         # 使用pre_tokenizer将输入文本分成pre_token列表
#     #         pre_token_list = pre_tokenize_text(text_str) 

#     #         for idx, pre_token in enumerate(pre_token_list):
#     #             new_pre_token, merged = find_and_merge_pair(pre_token, self.merges)
#     #             if not merged:
#     #                 break
#     #             pre_token_list[idx] = new_pre_token
#     #         text_split_by_specialToken[idx1].text_str = pre_token_list
#     #     # 4. 将每个pre_token转换成token id

#     #     for pre_token in text_split_by_specialToken:
#     #         for bytes_token in pre_token:
#     #             token_id = self.rev_vocab.get(bytes_token,-1)
#     #             if token_id == -1:
#     #                 raise ValueError(f"Token {bytes_token} not found in vocabulary.")
#     #             token_ids.append(token_id)
#     #     return token_ids
#     def encode(self, text: str) -> list[int]:
#         # 总任务: 将输入文本编码成 token id 列表

#         # 1. 先按照 special_tokens 切分文本
#         # text_split_by_specialToken 中的元素形如:
#         #   (text_str, is_special)
#         text_split_by_specialToken = split_text_by_special_tokens(
#             text,
#             self.special_tokens,
#         )

#         token_ids: list[int] = []

#         # 2. 依次处理每一段文本
#         for text_str, is_special in text_split_by_specialToken:
#             if is_special:
#                 # special token 直接按照 vocab 处理成一个 token id
#                 special_token_bytes = text_str.encode("utf-8")
#                 special_token_id = self.rev_vocab.get(special_token_bytes, -1)

#                 if special_token_id == -1:
#                     raise ValueError(
#                         f"Special token {text_str} not found in vocabulary."
#                     )

#                 token_ids.append(special_token_id)
#                 continue

#             # 3. 普通文本先经过 pre-tokenizer
#             # 假设 pre_tokenize_text 返回:
#             #   list[tuple[bytes, ...]]
#             pre_token_list = pre_tokenize_text(text_str)

#             # 4. 对每一个 pre_token，不断应用 merge，直到不能继续 merge
#             for idx, _ in enumerate(pre_token_list):
#                 while True:
#                     new_pre_token, merged = find_and_merge_pair(
#                         pre_token_list[idx],
#                         self.merges,
#                     )

#                     if not merged:
#                         break

#                     pre_token_list[idx] = new_pre_token

#                 # 5. 将最终的 bytes token 转换成 token id
#                 for bytes_token in pre_token_list[idx]:
#                     token_id = self.rev_vocab.get(bytes_token, -1)

#                     if token_id == -1:
#                         raise ValueError(
#                             f"Token {bytes_token} not found in vocabulary."
#                         )

#                     token_ids.append(token_id)

#         return token_ids

#     def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
#         # 挨个遍历iterable中的文本，编码成token id，并yield每个token id
#         for item in iterable:
#             token_ids = self.encode(item)
#             for token_id in token_ids:
#                 yield token_id

#     def decode(self, token_ids: list[int]) -> str:
#         # 将token id列表转换回文本
#         bytes_tokens = [self.vocab[token_id] for token_id in token_ids]
#         text = b"".join(bytes_tokens).decode("utf-8",errors="replace")
#         return text

# 把merge的逻辑改成dict，减少匹配数量
class BpeTokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.rev_vocab = {
            token_bytes: token_id
            for token_id, token_bytes in vocab.items()
        }

        self.merge_rank = {
            pair: rank
            for rank, pair in enumerate(merges)
        }

    def encode(self, text: str) -> list[int]:
        text_split_by_specialToken = split_text_by_special_tokens(
            text,
            self.special_tokens,
        )

        token_ids: list[int] = []

        for text_str, is_special in text_split_by_specialToken:
            if is_special:
                special_token_bytes = text_str.encode("utf-8")
                special_token_id = self.rev_vocab.get(special_token_bytes, -1)

                if special_token_id == -1:
                    raise ValueError(
                        f"Special token {text_str} not found in vocabulary."
                    )

                token_ids.append(special_token_id)
                continue

            pre_token_list = pre_tokenize_text(text_str)

            for pre_token in pre_token_list:
                while True:
                    pre_token, merged = find_and_merge_pair(
                        pre_token,
                        self.merge_rank,
                    )

                    if not merged:
                        break

                for bytes_token in pre_token:
                    token_id = self.rev_vocab.get(bytes_token, -1)

                    if token_id == -1:
                        raise ValueError(
                            f"Token {bytes_token} not found in vocabulary."
                        )

                    token_ids.append(token_id)

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for item in iterable:
            for token_id in self.encode(item):
                yield token_id

    def decode(self, token_ids: list[int]) -> str:
        bytes_tokens = [self.vocab[token_id] for token_id in token_ids]
        return b"".join(bytes_tokens).decode("utf-8", errors="replace")
    
# def find_and_merge_pair(
#     pre_token: tuple[bytes, ...],
#     merges: list[tuple[bytes, bytes]],
# ) -> tuple[tuple[bytes, ...], bool]:
#     """
#     给定一个 pre_token，例如：

#         (b'l', b'o', b'w', b'e', b'r')

#     和一组 merge 规则，例如：

#         [(b'l', b'o'), (b'lo', b'w'), ...]

#     找到第一个可以应用的 merge 规则，并对 pre_token 中所有匹配位置进行合并。

#     返回：
#         new_pre_token: 合并后的 pre_token
#         merged: 是否发生了合并
#     """

#     # 按 merges 的顺序找第一个能应用的规则
#     for merge in merges:
#         left, right = merge

#         # 先检查这个 merge 是否能在 pre_token 中找到
#         found = False
#         for i in range(len(pre_token) - 1):
#             if pre_token[i] == left and pre_token[i + 1] == right:
#                 found = True
#                 break

#         if not found:
#             continue

#         # 找到了可以应用的 merge 规则，开始真正合并
#         new_pre_token: list[bytes] = []
#         i = 0

#         while i < len(pre_token):
#             if (
#                 i < len(pre_token) - 1
#                 and pre_token[i] == left
#                 and pre_token[i + 1] == right
#             ):
#                 new_pre_token.append(left + right)
#                 i += 2
#             else:
#                 new_pre_token.append(pre_token[i])
#                 i += 1

#         return tuple(new_pre_token), True

#     # 所有 merge 规则都无法应用
#     return pre_token, False
def find_and_merge_pair(
    pre_token: tuple[bytes, ...],
    merge_rank: dict[tuple[bytes, bytes], int],
) -> tuple[tuple[bytes, ...], bool]:
    if len(pre_token) < 2:
        return pre_token, False

    best_idx = -1
    best_rank = float("inf")

    for i in range(len(pre_token) - 1):
        pair = (pre_token[i], pre_token[i + 1])
        rank = merge_rank.get(pair)

        if rank is not None and rank < best_rank:
            best_rank = rank
            best_idx = i

    if best_idx == -1:
        return pre_token, False

    merged_token = pre_token[best_idx] + pre_token[best_idx + 1]

    new_pre_token = (
        pre_token[:best_idx]
        + (merged_token,)
        + pre_token[best_idx + 2:]
    )

    return new_pre_token, True


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.
    """
    # 1. 读取训练数据
    train_data = read_corpus(input_path)

    # 2. 利用特殊词元标记边界
    boundaries = find_chunk_boundaries(train_data, split_special_tokens=special_tokens)

    # 3. 统计 pre-token 词频
    tokens_dic: dict[tuple[bytes, ...], int] = {}

    for start, end in boundaries:
        text = train_data[start:end].decode("utf-8")
        tokens_dic_temp = pre_tokenize(text)

        for token, count in tokens_dic_temp.items():
            tokens_dic[token] = tokens_dic.get(token, 0) + count

    # 4. 初始化 byte-level BPE 词表
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    merge_rule: list[tuple[bytes, bytes]] = []

    # 5. 把 tokens_dic 转成可变结构
    # words[word_id] 是当前 pre-token 的 BPE token 序列
    # word_freqs[word_id] 是这个 pre-token 的出现次数
    words: list[list[bytes]] = []
    word_freqs: list[int] = []

    for token, count in tokens_dic.items():
        words.append(list(token))
        word_freqs.append(count)

    # 6. 维护 pair 的全局频率，以及 pair 出现在哪些 word 中
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = {}

    for word_id, word in enumerate(words):
        freq = word_freqs[word_id]

        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])

            pair_counts[pair] = pair_counts.get(pair, 0) + freq

            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            pair_to_words[pair].add(word_id)

    # 7. 迭代训练 BPE
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # 7.1 选择最高频 pair
        # 频率相同的时候，选择字典序更大的 pair
        most_frequent_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], pair),
        )

        # 7.2 加入 merge_rule 和 vocab
        merge_rule.append(most_frequent_pair)
        vocab[len(vocab)] = most_frequent_pair[0] + most_frequent_pair[1]

        # 7.3 只更新包含 most_frequent_pair 的那些 word
        affected_word_ids = list(pair_to_words.get(most_frequent_pair, set()))

        for word_id in affected_word_ids:
            old_word = words[word_id]
            freq = word_freqs[word_id]

            # 先删除这个 old_word 对 pair_counts / pair_to_words 的旧贡献
            for i in range(len(old_word) - 1):
                old_pair = (old_word[i], old_word[i + 1])

                new_count = pair_counts.get(old_pair, 0) - freq
                if new_count > 0:
                    pair_counts[old_pair] = new_count
                else:
                    pair_counts.pop(old_pair, None)

                if old_pair in pair_to_words:
                    pair_to_words[old_pair].discard(word_id)
                    if not pair_to_words[old_pair]:
                        del pair_to_words[old_pair]

            # 在当前 word 中合并 most_frequent_pair
            new_word: list[bytes] = []
            i = 0

            while i < len(old_word):
                if (
                    i < len(old_word) - 1
                    and old_word[i] == most_frequent_pair[0]
                    and old_word[i + 1] == most_frequent_pair[1]
                ):
                    new_word.append(most_frequent_pair[0] + most_frequent_pair[1])
                    i += 2
                else:
                    new_word.append(old_word[i])
                    i += 1

            words[word_id] = new_word

            # 再加入 new_word 对 pair_counts / pair_to_words 的新贡献
            for i in range(len(new_word) - 1):
                new_pair = (new_word[i], new_word[i + 1])

                pair_counts[new_pair] = pair_counts.get(new_pair, 0) + freq

                if new_pair not in pair_to_words:
                    pair_to_words[new_pair] = set()
                pair_to_words[new_pair].add(word_id)

    return vocab, merge_rule
    
def merge_token(
    token: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    merged: list[bytes] = []
    i = 0

    while i < len(token):
        if (
            i < len(token) - 1
            and token[i] == pair[0]
            and token[i + 1] == pair[1]
        ):
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(token[i])
            i += 1

    return tuple(merged)

def read_corpus(input_path: str | os.PathLike) -> bytes:
    with open(input_path, "rb") as f:
        return f.read()

def find_chunk_boundaries(
    train_data: bytes,
    split_special_tokens: list[str],
) -> list[tuple[int, int]]:
    """
    Split train_data into chunks according to special tokens.

    Args:
        train_data:
            Corpus data read in binary mode, i.e. bytes.
        split_special_tokens:
            A list of special token strings. These special tokens are used as
            split markers and will not be included in returned chunks.

    Returns:
        A list of (start, end) byte offsets.
        Each range is [start, end), excluding special tokens.
    """

    if not isinstance(train_data, bytes):
        raise TypeError("train_data must be bytes. Open the file with 'rb'.")

    if not isinstance(split_special_tokens, list):
        raise TypeError("split_special_tokens must be a list[str].")

    if len(split_special_tokens) == 0:
        return [(0, len(train_data))] if len(train_data) > 0 else []

    split_tokens: list[bytes] = []

    for token in split_special_tokens:
        if not isinstance(token, str):
            raise TypeError("Each special token must be a str.")

        if token == "":
            raise ValueError("split_special_tokens cannot contain empty string.")

        split_tokens.append(token.encode("utf-8"))

    # 如果存在前缀关系，例如：
    # "<|end|>" 和 "<|endoftext|>"
    # 应该优先匹配更长的 special token。
    split_tokens.sort(key=len, reverse=True)

    boundaries: list[tuple[int, int]] = []

    chunk_start = 0
    i = 0
    n = len(train_data)

    while i < n:
        matched_token: Optional[bytes] = None

        for special_token in split_tokens:
            if train_data.startswith(special_token, i):
                matched_token = special_token
                break

        if matched_token is None:
            i += 1
            continue

        # 当前 normal text chunk 是 [chunk_start, i)
        # i 是 special token 的起始位置，所以 special token 不会被包含进去。
        if chunk_start < i:
            boundaries.append((chunk_start, i))

        # 跳过 special token
        i += len(matched_token)

        # 下一个 chunk 从 special token 后面开始
        chunk_start = i

    # 处理最后一段 normal text
    if chunk_start < n:
        boundaries.append((chunk_start, n))

    return boundaries

def split_text_by_special_tokens(
    text: str,
    special_tokens: list[str] | None = None,
) -> list[tuple[str, bool]]:
    """
    返回 [(片段, 是否为 special token), ...]
    """
    special_tokens = special_tokens or []

    if not special_tokens:
        return [(text, False)] if text else []

    # 长的 special token 放前面，处理 overlapping special tokens
    escaped_specials = [
        re.escape(tok)
        for tok in sorted(special_tokens, key=len, reverse=True)
    ]

    # 加括号，re.split 才会保留 special token 本身
    special_pat = "(" + "|".join(escaped_specials) + ")"

    parts = re.split(special_pat, text)
    special_set = set(special_tokens)

    result: list[tuple[str, bool]] = []

    for part in parts:
        if part == "":
            continue

        if part in special_set:
            result.append((part, True))
        else:
            result.append((part, False))

    return result

def pre_tokenize_text(text: str) -> list[tuple[bytes, ...]]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pre_tokens: list[tuple[bytes, ...]] = []

    for item in re.finditer(PAT, text):
        token = item.group()

        byte_tuple = tuple(
            bytes([b])
            for b in token.encode("utf-8")
        )

        pre_tokens.append(byte_tuple)

    return pre_tokens

def pre_tokenize(text: str) -> dict[tuple[bytes, ...], int]:

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens: dict[tuple[bytes, ...], int] = {}

    for item in re.finditer(PAT, text):
        token = item.group()

        # 把字符串 token 转成 UTF-8 bytes，然后每个 byte 单独作为一个初始 token
        # bpe 这里会把任何一种字符变成字节码(包括英文字母，汉字等)
        byte_tuple = tuple(bytes([b]) for b in token.encode("utf-8"))

        tokens[byte_tuple] = tokens.get(byte_tuple, 0) + 1

    return tokens
        
    