## Model

llama2-7b

## env

ipex 2.4.0 torch 2.4.0

## source code

from meta repo:https://github.com/meta-llama/llama

## error(error.txt)

File "/home/congxiao/.conda/envs/ipex/lib/python3.9/site-packages/intel_extension_for_pytorch/nn/utils/_weight_prepack.py", line 271, in forward
    output = torch.ops.torch_ipex.ipex_linear(
  File "/home/congxiao/.conda/envs/ipex/lib/python3.9/site-packages/torch/_ops.py", line 1061, in __call__
    return self_._op(*args, **(kwargs or {}))
RuntimeError: could not create a primitive descriptor for an inner product forward propagation primitive

## cpu(cpu.txt)

Intel(R) Xeon(R) Platinum 8458P(include amx extension)
