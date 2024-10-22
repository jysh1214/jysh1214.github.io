---
layout: post
title:  "Graph Break in TorchDynamo"
date:   2024-10-19 16:00:00 +0800
categories: [PyTorch]
author: Alex Chiang
---

**Environment**
- [Python 3.12.5](https://github.com/python/cpython/tree/v3.12.5)

Required Python packages:
- [torch v2.4.0](https://github.com/pytorch/pytorch/tree/v2.4.0)
- [iree-turbine 2.4.0](https://github.com/nod-ai/SHARK-Turbine)

To install the Python packages, run the following commands:
```bash
pip install torch==2.4.0
pip install iree-turbine==2.4.0
```


## What is Graph Break in TorchDynamo? ##

`Torchdynamo` will attempt to compile all of the torch/tensor operations within forward function into a single FX graph, but it may fail to capture everything into one graph.

When `TorchDynamo` encounters unsupported Python features, such as data-dependent control flow, it breaks the computation graph, lets the default Python interpreter handle the unsupported code, then resumes capturing the graph.


## Example Use Case: Control Flow ##

```python
import torch
import dis


class MyModule(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    if x.sum() > 0:
      return torch.cos(x)
    else:
      return torch.sin(x)


my_module = MyModule()
args = torch.rand(4)

opt_module = torch.compile(my_module, backend="turbine_cpu")

print(opt_module(args))
```

In the example below, we expected the computation to be split into 3 graphs, including:
- `x.sum()`
- `torch.cos(x)`
- `torch.sin(x)`


However, only 2 graphs were captured.

Graph code for `x.sum()`:
```python
def forward(self, L_x_: "f32[4][1]cpu"):
    l_x_ = L_x_

    sum_1: "f32[][]cpu" = l_x_.sum();  l_x_ = None
    gt: "b8[][]cpu" = sum_1 > 0;  sum_1 = None
    return (gt,)
```

Graph code for `torch.cos(x)`:
```python
def forward(self, L_x_: "f32[4][1]cpu"):
    l_x_ = L_x_
        
    cos: "f32[4][1]cpu" = torch.cos(l_x_);  l_x_ = None
    return (cos,)
```

It only recorded the operations executed with the inputs. This shows that the trace of a function depends on the inputs. Specifically, the trace is generated when the function is executed with actual arguments, not when `torch.compile` is written.


### Behavior with If-Else Branches ###

For the `if` branch, we should get graph 1 (`x.sum()`) and graph 2 (`torch.cos(x)`).
For the `else` branch, we should get graph 1 (`x.sum()`) and graph 3 (`torch.sin(x)`).

Here’s the execution of the model with both if and else conditions:
```python
import torch
import dis


class MyModule(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    if x.sum() > 0:
      return torch.cos(x)
    else:
      return torch.sin(x)


my_module = MyModule()
args = torch.rand(4)

opt_module = torch.compile(my_module, backend="turbine_cpu")

print(opt_module(args))  # for if condition
print(opt_module(-args)) # for else condition
```

![forward.subgraph](https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2024-10-18-Graph-Break-in-TorchDynamo/forward.subgraph.png?raw=true)

However, do we really need to recompile `graph 1` again? Since `graph 1` has already been compiled, reusing it should make more sense.


### Reusing Compiled Graph ###

`Guards` are a mechanism to check if a graph can be reused, so we don’t need to recompile it.

`Guards` check attributes of the inputs, such as device, shape, and stride of the input tensor. If these attributes remain the same, the compiled graph can be reused.

```
TREE_GUARD_MANAGER:
+- RootGuardManager
| +- DEFAULT_DEVICE: utils_device.CURRENT_DEVICE == None                           # _dynamo/output_graph.py:460 in init_ambient_guards
| +- GLOBAL_STATE: ___check_global_state()
| +- GuardManager: source=L['x'], accessed_by=DictGetItemGuardAccessor(x)
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CPU, BackendSelect, ADInplaceOrView, AutogradCPU), torch.float32, device=None, requires_grad=False, size=[4], stride=[1])  # if x.sum() > 0:  # in forward
| | +- NO_HASATTR: hasattr(L['x'], '_dynamo_dynamic_indices') == False           # if x.sum() > 0:  # in forward
```

### Graph Break Flow ###

![bytecode.transform](https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2024-10-18-Graph-Break-in-TorchDynamo/bytecode.transform.svg?raw=true)

The original bytecode of `forward` is:
```txt
9           0 RESUME                   0

12           2 LOAD_FAST                1 (x)
             4 LOAD_ATTR                1 (NULL|self + sum)
            24 CALL                     0
            32 LOAD_CONST               1 (0)
            34 COMPARE_OP              68 (>)
            38 POP_JUMP_IF_FALSE       21 (to 82)

13          40 LOAD_GLOBAL              3 (NULL + torch)
            50 LOAD_ATTR                4 (cos)
            70 LOAD_FAST                1 (x)
            72 CALL                     1
            80 RETURN_VALUE

15     >>   82 LOAD_GLOBAL              3 (NULL + torch)
            92 LOAD_ATTR                6 (sin)
           112 LOAD_FAST                1 (x)
           114 CALL                     1
           122 RETURN_VALUE
```

The transformed bytecode of `forward` is:
```txt
9           0 RESUME                   0
            2 PUSH_NULL
            4 LOAD_GLOBAL              8 (__compiled_fn_1)
           14 LOAD_FAST                1 (x)
           16 CALL                     1
           24 UNPACK_SEQUENCE          1
           28 POP_JUMP_IF_FALSE       12 (to 54)
           30 PUSH_NULL
           32 LOAD_GLOBAL             10 (__resume_at_40_2)
           42 LOAD_FAST                1 (x)
           44 CALL                     1
           52 RETURN_VALUE
      >>   54 PUSH_NULL
           56 LOAD_GLOBAL             12 (__resume_at_82_3)
           66 LOAD_FAST                1 (x)
           68 CALL                     1
           76 RETURN_VALUE
```

We can see two resume functions in the transformed bytecode because graph break occurred due to generic jump instruction.

When a graph break occurs in `TorchDynamo`, it generates a resume function to handle the part of the computation that couldn’t be captured in a single graph. This function resumes the execution from the point where the break happened. `TorchDynamo` generates new bytecode for the resume function, and that bytecode is then analyzed and compiled.

The generated bytecode of `__resume_at_40_2` is:
```txt
12           0 RESUME                   0
             2 JUMP_FORWARD            20 (to 44)
             4 RESUME                   0
             6 LOAD_FAST                0 (x)
             8 LOAD_ATTR                1 (NULL|self + sum)
            28 CALL                     0
            36 LOAD_CONST               1 (0)
            38 COMPARE_OP              68 (>)
            42 POP_JUMP_IF_FALSE       21 (to 86)

13     >>   44 LOAD_GLOBAL              3 (NULL + torch)
            54 LOAD_ATTR                4 (cos)
            74 LOAD_FAST                0 (x)
            76 CALL                     1
            84 RETURN_VALUE

15     >>   86 LOAD_GLOBAL              3 (NULL + torch)
            96 LOAD_ATTR                6 (sin)
           116 LOAD_FAST                0 (x)
           118 CALL                     1
           126 RETURN_VALUE
```

The transformed bytecode of `__resume_at_40_2` is:
```txt
12           0 RESUME                   0
             2 PUSH_NULL
             4 LOAD_GLOBAL              8 (__compiled_fn_5)
            14 LOAD_FAST                0 (x)
            16 CALL                     1
            24 UNPACK_SEQUENCE          1
            28 RETURN_VALUE
```

The compiled python code for the `GraphModule` `__compiled_fn_5` is:
```python
===== __compiled_fn_5 =====
def forward(self, L_x_: "f32[4][1]cpu"):
    l_x_ = L_x_

    cos: "f32[4][1]cpu" = torch.cos(l_x_);  l_x_ = None
    return (cos,)
```

After compiling the graph, `TorchDynamo` generates `Guards` based on the bytecode and the input attributes to determine whether the graph can be reused in future executions. 

Similarly, `__resume_at_82_3` follows the same process.

The transformed bytecode of `__resume_at_82_3` is:
```txt
12           0 RESUME                   0
             2 PUSH_NULL
             4 LOAD_GLOBAL              8 (__compiled_fn_7)
            14 LOAD_FAST                0 (x)
            16 CALL                     1
            24 UNPACK_SEQUENCE          1
            28 RETURN_VALUE
```

The compiled python code for `__compiled_fn_7` is:
```python
===== __compiled_fn_7 =====
def forward(self, L_x_: "f32[4][1]cpu"):
    l_x_ = L_x_

    sin: "f32[4][1]cpu" = torch.sin(l_x_);  l_x_ = None
    return (sin,)
```

### IREE-Turbine Backend ###

After being optimized by the `IREE-Turbine` backend, the compiled graphs are transformed into MLIR for efficient execution. Below are the MLIR representations for each graph.

**Graph 1**:

```mlir
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<() -> ()>
module {
  util.func public @main$async(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> !hal.buffer_view attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import wait(%arg1) => %arg0 : !hal.buffer_view -> tensor<4xf32>
    %1 = tensor.empty() : tensor<f32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<f32>) -> tensor<f32>
    %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%0 : tensor<4xf32>) outs(%2 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.addf %in, %out : f32
      linalg.yield %8 : f32
    } -> tensor<f32>
    %4 = tensor.empty() : tensor<i1>
    %5 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%3 : tensor<f32>) outs(%4 : tensor<i1>) {
    ^bb0(%in: f32, %out: i1):
      %8 = arith.cmpf ogt, %in, %cst : f32
      linalg.yield %8 : i1
    } -> tensor<i1>
    %6 = hal.tensor.barrier join(%5 : tensor<i1>) => %arg2 : !hal.fence
    %7 = hal.tensor.export %6 : tensor<i1> -> !hal.buffer_view
    util.return %7 : !hal.buffer_view
  }
  util.func public @main(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = util.null : !hal.fence
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    %1 = util.call @main$async(%arg0, %0, %fence) : (!hal.buffer_view, !hal.fence, !hal.fence) -> !hal.buffer_view
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.return %1 : !hal.buffer_view
  }
}
```

**Graph 2**:

```mlir
#map = affine_map<(d0) -> (d0)>
module {
  util.func public @main$async(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> !hal.buffer_view attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
    %0 = hal.tensor.import wait(%arg1) => %arg0 : !hal.buffer_view -> tensor<4xf32>
    %1 = tensor.empty() : tensor<4xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<4xf32>) outs(%1 : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = math.cos %in : f32
      linalg.yield %5 : f32
    } -> tensor<4xf32>
    %3 = hal.tensor.barrier join(%2 : tensor<4xf32>) => %arg2 : !hal.fence
    %4 = hal.tensor.export %3 : tensor<4xf32> -> !hal.buffer_view
    util.return %4 : !hal.buffer_view
  }
  util.func public @main(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = util.null : !hal.fence
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    %1 = util.call @main$async(%arg0, %0, %fence) : (!hal.buffer_view, !hal.fence, !hal.fence) -> !hal.buffer_view
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.return %1 : !hal.buffer_view
  }
}
```

**Graph 3**:

```mlir
#map = affine_map<(d0) -> (d0)>
module {
  util.func public @main$async(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> !hal.buffer_view attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
    %0 = hal.tensor.import wait(%arg1) => %arg0 : !hal.buffer_view -> tensor<4xf32>
    %1 = tensor.empty() : tensor<4xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<4xf32>) outs(%1 : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = math.sin %in : f32
      linalg.yield %5 : f32
    } -> tensor<4xf32>
    %3 = hal.tensor.barrier join(%2 : tensor<4xf32>) => %arg2 : !hal.fence
    %4 = hal.tensor.export %3 : tensor<4xf32> -> !hal.buffer_view
    util.return %4 : !hal.buffer_view
  }
  util.func public @main(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = util.null : !hal.fence
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    %1 = util.call @main$async(%arg0, %0, %fence) : (!hal.buffer_view, !hal.fence, !hal.fence) -> !hal.buffer_view
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.return %1 : !hal.buffer_view
  }
}
```

## Debugging ##

### Decompile Tools ###

We successfully generated 3 graphs as expected, but how do we know what the `__resume_at_40_2` and `__resume_at_82_3` correspond to?

To understand the bytecode generated by `torch.compile`, tools like `depyf` can be helpful for decompiling the bytecode back into human-readable form.

To install `depyf`, use the following command:
```bash
pip install depyf
```

To use `depyf` in your code, you need to install its hooks with the following command:
```python
import depyf
depyf.install()
```

After installation, you can decompile the bytecode generated by `torch.compile`. Below are examples of decompiled bytecode for the generated graphs.

The transformed bytecode of `forward`:
```txt
  9           0 RESUME                   0
              2 PUSH_NULL
              4 LOAD_GLOBAL              8 (__compiled_fn_1)
             14 LOAD_FAST                1 (x)
             16 CALL                     1
             24 UNPACK_SEQUENCE          1
             28 POP_JUMP_IF_FALSE       12 (to 54)
             30 PUSH_NULL
             32 LOAD_GLOBAL             10 (__resume_at_40_2)
             42 LOAD_FAST                1 (x)
             44 CALL                     1
             52 RETURN_VALUE
        >>   54 PUSH_NULL
             56 LOAD_GLOBAL             12 (__resume_at_82_3)
             66 LOAD_FAST                1 (x)
             68 CALL                     1
             76 RETURN_VALUE
```

Possible source code:
```python
def forward(self, x):
    __temp_2, = __compiled_fn_1(x)
    if __temp_2:
        return __resume_at_40_2(x)
    return __resume_at_82_3(x)
```

The transformed bytecode of `__resume_at_40_2`:
```txt
12           0 RESUME                   0
             2 PUSH_NULL
             4 LOAD_GLOBAL              8 (__compiled_fn_5)
            14 LOAD_FAST                0 (x)
            16 CALL                     1
            24 UNPACK_SEQUENCE          1
            28 RETURN_VALUE
```

Possible source code:
```python
def torch_dynamo_resume_in_forward_at_12(x):
    __temp_6, = __compiled_fn_5(x)
    return __temp_6
```

The transformed bytecode of `__resume_at_82_3`:
```txt
12           0 RESUME                   0
             2 PUSH_NULL
             4 LOAD_GLOBAL              8 (__compiled_fn_7)
            14 LOAD_FAST                0 (x)
            16 CALL                     1
            24 UNPACK_SEQUENCE          1
            28 RETURN_VALUE
```

Possible source code:
```python
def torch_dynamo_resume_in_forward_at_12(x):
    __temp_8, = __compiled_fn_7(x)
    return __temp_8
```

### TorchDynamo Explain ###

Minimizing graph breaks is crucial for maintaining efficiency in `TorchDynamo`. Graph breaks can interrupt the flow of optimized execution, leading to potential performance degradation. To help diagnose and understand the causes of graph breaks, you can use tools like `torch._dynamo.explain()` and the environment variable `TORCH_LOGS=graph_breaks`. These tools help developers identify and resolve issues in their code to reduce graph breaks and improve performance.

Here’s how you can explain the causes of graph breaks:

```
args = torch.rand(4)

explanation = torch._dynamo.explain(my_module)(args)
print(explanation)
```

Result:
```txt
Graph Count: 2
Graph Break Count: 1
Op Count: 2
Break Reasons:
  Break Reason 1:
    Reason: generic_jump TensorVariable()
    User Stack:
      <FrameSummary file sub_graph.py, line 12 in forward>
Ops per Graph:
  Ops 1:
    <built-in function gt>
  Ops 2:
    <built-in method cos of type object at 0x107958cf0>
Out Guards:
  Guard 1:
    Name: ''
    Source: shape_env
    Create Function: SHAPE_ENV
    Guard Types: None
    Code List: None
    Object Weakref: None
    Guarded Class Weakref: None
    ...
```

If you want to strictly enforce full graph execution and throw an error upon encountering the first graph break, you can disable Python fallbacks by setting `fullgraph=True` in `torch.compile()`:
```python
opt_module = torch.compile(my_module, fullgraph=True, backend="turbine_cpu")
```

## Summary ##

In this post, we introduce the two key point:
- `TorchDynamo` breaks the graph when it encounters unsupported Python features.
- `Guards` help reuse compiled graphs to avoid recompilation.


## See Also ##

- [A Walkthrough Example of torch.compile with IREE-Turbine](https://jysh1214.github.io/pytorch/2024/10/08/A-Walkthrough-Example-of-torch.compile-with-IREE-Turbine.html)
- [iree](https://iree.dev)


## References ##

- [Dynamo Deep-Dive](https://pytorch.org/docs/stable/torch.compiler_dynamo_deepdive.html#id1)
- [Introduction to torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#torchdynamo-and-fx-graphs)
- [A Walk Through Example of torch.compile](https://depyf.readthedocs.io/en/latest/walk_through.html)
