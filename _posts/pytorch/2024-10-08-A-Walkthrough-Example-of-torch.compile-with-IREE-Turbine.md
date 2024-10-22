---
layout: post
title:  "A Walkthrough Example of torch.compile with IREE-Turbine"
date:   2024-10-08 16:00:00 +0800
categories: [PyTorch]
author: Alex Chiang
---

In this post, we’ll show some demo code that you can easily reproduce with the following environment setup.

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


## What is torch.compile? ##

`torch.compile` is a JIT (Just-In-time) compiler that optimizes arbitrary Python functions or models by using a specified backend.

### Basic Usage ###

To optimize a `torch.nn.Module`, we need to wrap it with a `torch.compile` call.

```python
import torch

class LinearModule(torch.nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
    self.bias = torch.nn.Parameter(torch.randn(out_features))

  def forward(self, input):
    return (input @ self.weight) + self.bias

linear_module = LinearModule(4, 3)

# Compile the module
optimized_linear_module = torch.compile(linear_module)
```


## How does torch.compile work? ##

![TorchCompile](https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2024-10-08-A-Walkthrough-Example-of-torch.compile-with-IREE-Turbine/pytorch-2.0-img12.png?raw=true)

### TorchDynamo ###

`TorchDynamo` is responsible for JIT compiling arbitrary Python code into FX graphs, which can then be further optimized.

`TorchDynamo` will capture the bytecode before PVM execute them using Frame Evaluation API ([PEP-0523](https://peps.python.org/pep-0523/)) and then extracts FX graphs by analyzing the bytecode during runtime and detecting calls to PyTorch operations.

![TorchDynamo](https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2024-10-08-A-Walkthrough-Example-of-torch.compile-with-IREE-Turbine/torch-dynamo.png?raw=true)

### FX ###

FX is a toolkit for developers to use to transform `torch.nn.Module` instances. FX consists of three main components: a symbolic tracer, an intermediate representation, and Python code generation.

```python
import torch

class LinearModule(torch.nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
    self.bias = torch.nn.Parameter(torch.randn(out_features))

  def forward(self, input):
    return (input @ self.weight) + self.bias

linear_module = LinearModule(4, 3)

from torch.fx import symbolic_trace
# Symbolic tracing frontend - captures the semantics of the module
symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)

# FX Graph IR - Graph representation
print(symbolic_traced.graph)

# Code generation - valid Python code matching the graph
print(symbolic_traced.code)
```

The FX graph IR (`torch.fx.Graph`):
```txt
graph():
    %input_1 : [num_users=1] = placeholder[target=input]
    %weight : [num_users=1] = get_attr[target=weight]
    %matmul : [num_users=1] = call_function[target=operator.matmul](args = (%input_1, %weight), kwargs = {})
    %bias : [num_users=1] = get_attr[target=bias]
    %add : [num_users=1] = call_function[target=operator.add](args = (%matmul, %bias), kwargs = {})
    return add
```

The Python code:
```python
def forward(self, input):
    input_1 = input
    weight = self.weight
    matmul = input_1 @ weight;  input_1 = weight = None
    bias = self.bias
    add = matmul + bias;  matmul = bias = None
    return add
```

### AOTAutograd ###

`AOTAutograd` will generate backward computation graph from forward computation graph.

---

### A Walkthrough of the Example ###

In this section, we will use the following as an example and focus on how `torch.compile` optimize inference processing and how it works with `torch.nn.Module`.

```python
import torch

class LinearModule(torch.nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
    self.bias = torch.nn.Parameter(torch.randn(out_features))

  def forward(self, input):
    return (input @ self.weight) + self.bias

linear_module = LinearModule(4, 3)

# To optimize the module using `iree-turbine` backend,
# we need to specify the `turbine_cpu` backend.
opt_linear_module = torch.compile(linear_module, backend="turbine_cpu")

args = torch.randn(4)
turbine_output = opt_linear_module(args)
print(turbine_output)
```

The backend function will be `shark_turbine/dynamo/backends/cpu.py(45)_base_backend()`:
```python
def _base_backend(gm: torch.fx.GraphModule, example_inputs):
    # Set up the session, context and invocation.
    # Note that we do this on one in-memory module in a few phases:
    #  1. Build it from the FX graph.
    #  2. Run torch MLIR passes to lower it to a suitable form for
    #     input.
    #  3. Run IREE's main compiler.
    #  4. Output to an mmap buffer.
    session = Session()
    session.set_flags(*DEFAULT_COMPILER_FLAGS)
    session.set_flags("--iree-hal-target-backends=llvm-cpu")
    context = session.context
    importer = FxImporter(context=context)
    module = importer.module
    inv = session.invocation()
    # TODO: Should capture diagnostics.
    inv.enable_console_diagnostics()
    inv.import_module(module.operation)

    # Apply decompositions.
    gm = turbine_cpu_pass_pipeline(gm, example_inputs)

    # Import phase.
    importer.import_graph_module(gm)
    print(module, file=sys.stderr)
    with context:
        pm = PassManager.parse("builtin.module(torch-to-iree)")
        pm.run(module.operation)
    print(module, file=sys.stderr)

    # IREE compilation phase.
    inv.execute()

    # Output phase.
    output = Output.open_membuffer()
    inv.output_vm_bytecode(output)

    # Set up for runtime.
    device_state = _get_device_state()
    # TODO: Switch to wrap_buffer once https://github.com/openxla/iree/issues/14926
    # is fixed.
    # vmfb_module = VmModule.wrap_buffer(
    #     device_state.instance,
    #     output.map_memory(),
    #     destroy_callback=output.close,
    # )
    vmfb_module = VmModule.copy_buffer(
        device_state.instance,
        output.map_memory(),
    )
    output.close()

    return SpecializedExecutable(vmfb_module, device_state)
```

`TorchDynamo` will dynamically analyze the bytecode, transform it into an FX graph by generating corresponding FX nodes based on the bytecode instructions.

The captured bytecode from `linear_module.forward()`:
```txt
 11           0 RESUME                   0
 12           2 LOAD_FAST                1 (input)
              4 LOAD_FAST                0 (self)
              6 LOAD_ATTR                0 (weight)
             26 BINARY_OP                4 (@)
             30 LOAD_FAST                0 (self)
             32 LOAD_ATTR                2 (bias)
             52 BINARY_OP                0 (+)
             56 RETURN_VALUE
```

> *NOTE*
> To dump the bytecode, you can use the `dis` module:
> ```python
> import dis
> 
> dis.dis(linear_module.forward)
> ```

The generated FX graph (`torch.fx.Graph`) will be used to create a new FX GraphModule (`torch.fx.GraphModule`).
We can use `graph()` method to dump the FX graph IR of the GraphModule.
```txt
graph():
    %primals_1 : [num_users=1] = placeholder[target=primals_1]
    %primals_2 : [num_users=1] = placeholder[target=primals_2]
    %primals_3 : [num_users=1] = placeholder[target=primals_3]
    %unsqueeze : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_3, 0), kwargs = {})
    %mm : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%unsqueeze, %primals_1), kwargs = {})
    %squeeze : [num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%mm, 0), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze, %primals_2), kwargs = {})
    return [add, unsqueeze]
```

The new FX GraphModule will use `recompile()` method to generate valid Python code matching the graph. 
```python
def forward(self, primals_1: "f32[4, 3][3, 1]cpu", primals_2: "f32[3][1]cpu", primals_3: "f32[4][1]cpu"):
    unsqueeze: "f32[1, 4][4, 1]cpu" = torch.ops.aten.unsqueeze.default(primals_3, 0);  primals_3 = None
    mm: "f32[1, 3][3, 1]cpu" = torch.ops.aten.mm.default(unsqueeze, primals_1);  primals_1 = None
    squeeze: "f32[3][1]cpu" = torch.ops.aten.squeeze.dim(mm, 0);  mm = None
    add: "f32[3][1]cpu" = torch.ops.aten.add.Tensor(squeeze, primals_2);  squeeze = primals_2 = None
    return [add, unsqueeze]
```

`AOTAutograd` will generate backward computation graph from forward graph. In this post, we will focus on the forward function.
```python
def forward(self, primals_1: "f32[4, 3][3, 1]cpu", primals_2: "f32[3][1]cpu", primals_3: "f32[4][1]cpu"):
    unsqueeze: "f32[1, 4][4, 1]cpu" = torch.ops.aten.unsqueeze.default(primals_3, 0);  primals_3 = None
    mm: "f32[1, 3][3, 1]cpu" = torch.ops.aten.mm.default(unsqueeze, primals_1);  primals_1 = None
    squeeze: "f32[3][1]cpu" = torch.ops.aten.squeeze.dim(mm, 0);  mm = None
    add: "f32[3][1]cpu" = torch.ops.aten.add.Tensor(squeeze, primals_2);  squeeze = primals_2 = None
    return [add, unsqueeze]

def backward(self, unsqueeze: "f32[1, 4][4, 1]cpu", tangents_1: "f32[3][1]cpu"):
    unsqueeze_1: "f32[1, 3][3, 1]cpu" = torch.ops.aten.unsqueeze.default(tangents_1, 0)
    t: "f32[4, 1][1, 4]cpu" = torch.ops.aten.t.default(unsqueeze);  unsqueeze = None
    mm_1: "f32[4, 3][3, 1]cpu" = torch.ops.aten.mm.default(t, unsqueeze_1);  t = unsqueeze_1 = None
    return [mm_1, tangents_1, None]
```

The `torch.fx.GraphModule` of the `forward` function will be passed to the `iree-turbine` backend function.
We can use `print_readable()` method to show the Python code matching the graph in the backend function:
```python
# gm.print_readable(include_stride=True, include_device=True)
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 3][3, 1]cpu", primals_2: "f32[3][1]cpu", primals_3: "f32[4][1]cpu"):
        unsqueeze: "f32[1, 4][4, 1]cpu" = torch.ops.aten.unsqueeze.default(primals_3, 0);  primals_3 = None
        mm: "f32[1, 3][3, 1]cpu" = torch.ops.aten.mm.default(unsqueeze, primals_1);  primals_1 = None
        squeeze: "f32[3][1]cpu" = torch.ops.aten.squeeze.dim(mm, 0);  mm = None
        add: "f32[3][1]cpu" = torch.ops.aten.add.Tensor(squeeze, primals_2);  squeeze = primals_2 = None
        return [add, unsqueeze]

# Remove stride and device to make the code clearer
# gm.print_readable()
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 3]", primals_2: "f32[3]", primals_3: "f32[4]"):
        unsqueeze: "f32[1, 4]" = torch.ops.aten.unsqueeze.default(primals_3, 0);  primals_3 = None
        mm: "f32[1, 3]" = torch.ops.aten.mm.default(unsqueeze, primals_1);  primals_1 = None
        squeeze: "f32[3]" = torch.ops.aten.squeeze.dim(mm, 0);  mm = None
        add: "f32[3]" = torch.ops.aten.add.Tensor(squeeze, primals_2);  squeeze = primals_2 = None
        return [add, unsqueeze]
```

---

#### IREE-Turbine Backend ####

`iree` is a MLIR-based compiler and it cannot handle FX GraphModule directly.
In the backend function, we use `torch-mlir` FX Importer to convert the FX GraphModule into the Torch MLIR dialect.

![What is Torch-MLIR?](https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2024-10-08-A-Walkthrough-Example-of-torch.compile-with-IREE-Turbine/torch-mlir.png?raw=true)

`torch-mlir/python/torch_mlir/extras/fx_importer.py`:
```python
class FxImporter:
    """Main entry-point for importing an fx.GraphModule.

    The FxImporter is a low-level class intended for framework integrators.
    It provides several options for customization:

    * config_check: Optionally allows some per-import configuration safety
      checks to be skipped.
    * literal_resolver_callback: Callback that will be invoked when a literal,
      live torch.Tensor is encountered in the FX graph, allowing the default
      action (which is to inline the data as a DenseResourceElementsAttr) to
      be completely overriden.
    * py_attr_tracker: Weak reference tracker for live PyTorch objects used
      to unique them with respect to attributes. If not specified, there will
      be one reference tracker per import, but this can be injected to share
      the same uniqueing across imports (i.e. if building multiple functions
      into the same context or module).
    """
    # ...
```

The MLIR converted from the FX GraphModule is:
```mlir
module {
  func.func @main(%arg0: !torch.vtensor<[4,3],f32>, %arg1: !torch.vtensor<[3],f32>, %arg2: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[3],f32>, !torch.vtensor<[1,4],f32>) {
    %int0 = torch.constant.int 0
    %0 = torch.aten.unsqueeze %arg2, %int0 : !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[1,4],f32>
    %1 = torch.aten.mm %0, %arg0 : !torch.vtensor<[1,4],f32>, !torch.vtensor<[4,3],f32> -> !torch.vtensor<[1,3],f32>
    %int0_0 = torch.constant.int 0
    %2 = torch.aten.squeeze.dim %1, %int0_0 : !torch.vtensor<[1,3],f32>, !torch.int -> !torch.vtensor<[3],f32>
    %int1 = torch.constant.int 1
    %3 = torch.aten.add.Tensor %2, %arg1, %int1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[3],f32>
    return %3, %0 : !torch.vtensor<[3],f32>, !torch.vtensor<[1,4],f32>
  }
}
```

After applying the lowering pass using the `iree` compiler with the following code:
```python
pm = PassManager.parse("builtin.module(torch-to-iree)")
pm.run(module.operation)
```

The MLIR will be:
```mlir
#map = affine_map<(d0) -> (d0)>
module {
  util.func public @main$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> (!hal.buffer_view, !hal.buffer_view) attributes {inlining_policy = #util.inline.never, iree.abi.model = "coarse-fences", iree.abi.stub} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<4x3xf32>
    %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<3xf32>
    %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<4xf32>
    %expanded = tensor.expand_shape %2 [[0, 1]] output_shape [1, 4] : tensor<4xf32> into tensor<1x4xf32>
    %3 = tensor.empty() : tensor<1x3xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %5 = linalg.matmul ins(%expanded, %0 : tensor<1x4xf32>, tensor<4x3xf32>) outs(%4 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %collapsed = tensor.collapse_shape %5 [[0, 1]] : tensor<1x3xf32> into tensor<3xf32>
    %6 = tensor.empty() : tensor<3xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%collapsed, %1 : tensor<3xf32>, tensor<3xf32>) outs(%6 : tensor<3xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %11 = arith.addf %in, %in_0 : f32
      linalg.yield %11 : f32
    } -> tensor<3xf32>
    %8:2 = hal.tensor.barrier join(%7, %expanded : tensor<3xf32>, tensor<1x4xf32>) => %arg4 : !hal.fence
    %9 = hal.tensor.export %8#0 : tensor<3xf32> -> !hal.buffer_view
    %10 = hal.tensor.export %8#1 : tensor<1x4xf32> -> !hal.buffer_view
    util.return %9, %10 : !hal.buffer_view, !hal.buffer_view
  }
  util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view) attributes {iree.abi.stub} {
    %0 = util.null : !hal.fence
    %c-1_i32 = arith.constant -1 : i32
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    %1:2 = util.call @main$async(%arg0, %arg1, %arg2, %0, %fence) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> (!hal.buffer_view, !hal.buffer_view)
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.return %1#0, %1#1 : !hal.buffer_view, !hal.buffer_view
  }
}
```

> *NOTE*
> To reproduce the result above, you can use the `iree-opt` tool with the following command:
> ```bash
> iree-opt --pass-pipeline="builtin.module(torch-to-iree)" <your_mlir>
> ```

The `iree` compiler will compile the MLIR into the VMFB (VM Frame Buffer) format, which can be executed by the `iree` runtime. The related code is:
```python
# IREE compilation phase.
inv.execute()

# Output phase.
output = Output.open_membuffer()
inv.output_vm_bytecode(output)
```

Finally, we will pass the args to the `iree` runtime and invoke it.
```python
class SpecializedExecutable:
    """A concrete executable that has been specialized in some way."""
    #...

    def __call__(self, *inputs):
        arg_list = VmVariantList(len(inputs))
        ret_list = VmVariantList(
            1
        )  # TODO: Get the number of results from the descriptor.

        # Move inputs to the device and add to arguments.
        self._inputs_to_device(inputs, arg_list)
        # TODO: Append semaphores for async execution.

        # Invoke.
        self.vm_context.invoke(self.entry_function, arg_list, ret_list)
        return self._returns_to_user(ret_list)
```

The answer will look like this:
```python
tensor([ 1.1014,  0.9572, -0.8066], grad_fn=<CompiledFunctionBackward>)
```

We can summarize the basic data flow for each phase of the process, from Python code to MLIR.
![TorchCompileDataFlow](https://github.com/jysh1214/jysh1214.github.io/blob/master/_assets/2024-10-08-A-Walkthrough-Example-of-torch.compile-with-IREE-Turbine/torch-compile-data-flow.png?raw=true)


## Summary ##

In this blog, we dive into how to use `torch.compile` with the `IREE-Turbine` backend to optimize and execute PyTorch models. The blog provides a step-by-step guide that walks you through the entire data flow, from Python source code to MLIR.

Although `torch.compile` is simple to use, it feels like a black box, making it difficult to understand what happens during the optimization process.

In future blogs, I plan to delve deeper into `TorchDynamo` and `FX`, offering more detailed explanations and introducing methods for probing and debugging to help understand these processes.


## See Also ##

- [iree](https://iree.dev)
- [torch-mlir](https://github.com/llvm/torch-mlir/tree/main)


## References ##

- [Introduction to torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html#torchdynamo-and-fx-graphs)
- [torch.fx](https://pytorch.org/docs/stable/fx.html)
