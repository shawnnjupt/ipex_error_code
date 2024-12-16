# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from generation import Llama
from typing import List
import intel_extension_for_pytorch  as ipex

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    model = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    model = ipex.optimize(model.eval(),dtype=torch.bfloat16)

    prompts: List[str] = ["huggingface is an company"] 
    results = model.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for prompt, result in zip(prompts, results):
        print("\n==================================\n")
        print(prompt)
        print(result['generation'])
        print("\n==================================\n")


if __name__ == "__main__":
    # _enable_tpp
    checkpoints_dir='/home/congxiao/model/llama/llama-2-7b/'
    tokenizer_path='/home/congxiao/model/llama/llama-2-7b/tokenizer.model'
    max_seq_len=1024
    max_gen_len=1
    main(ckpt_dir=checkpoints_dir,
         tokenizer_path=tokenizer_path,
         max_seq_len=max_seq_len,
         max_gen_len=max_gen_len
         )
