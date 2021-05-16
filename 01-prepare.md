### Prepare流程

***

> 工欲善其事必先利其器 -- 《论语·卫灵公》

***

**本文基于Pytorch版本v1.8.0 (@37c1f4)。**

熟悉QAT流程的读者肯定知道，如果希望对一个模型进行QAT量化，一般需要对网络中的权重（weight）和激活值（activation）进行模拟量化，并在训练中借助模拟量化节点反向传播，对网络进行训练。

一个普通的nn.Module是如何转换得到插入模拟量化节点的模型的呢？在FX中，这个转换过程大体分为四个步骤：

（1）**符号追踪（symbolic trace）**：获得原始nn.Module网络的图表示

（2）**模块融合（fuse）**: 此步骤将满足模式的相邻模块融合成nn.Sequential。例如，将相邻的nn.Conv2d和nn.BatchNorm2d打包到一个nn.Sequential内，构成一个FusedModule，此后这个FusedModule将在prepare环节被替换成一个融合的模块。例如nn.Conv2d+nn.BatchNorm2d -> nn.intrinsic.qat.ConvBN2d。

（3）**模块交换（swap）**：此步骤是为了实现weight的模拟量化。由于nn.Conv2d中的weight是不带模拟量化（Fake Quantize）功能的，为了实现QAT中的模拟量化，需要把所有普通模块替换成可以进行模拟量化的QAT module，而这些module一般都定义在torch.nn.qat当中。torch官方已经对这些模拟量化module的forward和backward进行了实现，可以在训练过程中实现反向传播。例如nn.Conv2d -> nn.qat.intrinsic.Conv2d。

（4）**activation量化节点插入（insert observer）**：此步骤是为了实现对activation的模拟量化（或者在PTQ中插入观察节点，实现对激活值数据分布的观察）。

以上即为一个普通的nn.Module转换为一个可进行QAT的graph module的大致流程。本文后续内容将深入每个环节对其中包含的诸多操作进行拆解，条分缕析。

***

0. **预备内容**
   
FX中的核心数据结构（包括量化相关的部分）目前基本分布在torch/fx和torch/quantization两个文件夹下：
   - Node：在torch/fx/node.py
   - Graph：在
   - GraphModule：在
   - FakeQuantize：在torch/quantization/fake_quantize
   - Observer：
   - QConfig：QConfig中定义了weight和activation的量化方式

1. **QAT Prepare**

```python
prepared = quantizer.prepare(graph_module, 
                             qconfig_dict,
                             tracer.node_name_to_scope,
                             prepare_custom_config_dict=prepare_custom_config_dict,
                             is_standalone_module=is_standalone_module
                            )
```

此处即prepare的入口。


3. **模块交换（swap）**

**3.1 准备工作 —— 传播量化配置（propagate）**
    
在实际量化中，网络中各个层的量化配置（权重位宽、激活值位宽等）未必完全相同。为此FX允许用户定义

```python
propagate_qconfig_(model, flattened_qconfig_dict)
```

**3.2 模块替换**

在传播完成量化配置后，可以对每个模块

```python
if model.training:
   additional_qat_module_mapping = prepare_custom_config_dict.get(
                "additional_qat_module_mapping", {})
   self._qat_swap_modules(model, additional_qat_module_mapping)
```

我们来看一下_qat_swap_modules函数的内容:
```python

def _qat_swap_modules(
        self, root: torch.nn.Module,
        additional_qat_module_mapping: Dict[Callable, Callable]) -> None:
        all_mappings = get_combined_dict(
            get_default_qat_module_mappings(), additional_qat_module_mapping)
    convert(root, mapping=all_mappings, inplace=True, remove_qconfig=False)

```

_qat_swap_module函数先对默认的qat_module_mapping和用户提供的addtional_qat_mappiing方式进行合并，得到网络整体的QAT module的映射方式，之后调用convert函数，将graph module和mapping方式一并作为参数传入，开始进行替换。

随着调用栈的深入，我们可以发现convert最终调用了torch/quantization/quantize.py中的_convert函数。

为简便起见，我们跳过_convert函数中参数准备的部分，直入正题，看看它是如何实现module到QAT module替换的。

```python

for name, mod in module.named_children():
    # both fused modules and observed custom modules are
    # swapped as one unit
    if not isinstance(mod, _FusedModule) and \
       type(mod) not in custom_module_class_mapping:
        _convert(mod, mapping, True,  # inplace
                 custom_module_class_mapping)
    reassign[name] = swap_module(mod, mapping,custom_module_class_mapping)
for key, value in reassign.items():
    module._modules[key] = value
return module

```

以上就是其中替换部分的代码片段，可以看出，此处使用了一个递归实现，当传入该函数的mod不是一个FusedModule，且这个mod没有相应的用户自定义mapping时，就递归执行。当mod满足其中任意一个条件时，则执行swap_module。而swap_module所做的正是把传入的mod根据mapping包含的映射关系，获得其对应的QAT module。

当然这一步并不会发生真正意义上的替换，仅仅是把各个mod对应的QAT module根据mod的名字放到一个字典（dict）当中。当网络的所有mod都被递归地映射完成，得到所有的[mod name, QAT module]的关系后，再统一进行更新。这里就用到了Pytorch nn.Module的保存模块的一个特性，也即nn.Module所包含的module实际都存储在_modules这个字典当中，通过对这个字典的键值对进行更新就实现了对nn.Module的更新。

以上即是QAT module的自动替换的过程。

对普通nn.Module与QAT module在Pytorch中区别感兴趣的读者，我们在下面一小节中单独进行了一些介绍，而对于主要想了解FX工作原理的读者，下面一小节可以跳过，完全不影响后续的阅读。

**3.3 nn.Module，QAT module，quantized module的区别**

3.3.1 nn.Module

普通的nn.Module实现在torch/nn/modules下，其中都是我们耳熟能详的一些网络操作，例如Conv1d/2d/3d，Linear，BatchNorm1d/2d/3d，ReLU/Sigmoid/Swish等等。这些模块的forward实际调用了torch.nn.functional中对应的可微分函数，因此可以在反向传播中被优化。

3.3.2 QAT module

QAT module实现在

3.3.3 quantized module

quantized module实现在torch\nn\qat\modules和torch\nn\intrinsic\qat\modules。

其中intrinsic下的模块均为fused后的操作，目前包含了：
- **ConvBN（1d/2d/3d）**
- **ConvBNReLu（1d/2d/3d）**
- **LinearReLU**

而torch/nn/qat/modules下的模块与普通nn.Module对应，均为独立的操作。但由于并非所有的nn.Module都存在可量化的实现方式，所以qat/modules下的模块目前都是


4. **activation量化节点插入**

以上我们看到了FX是如何实现对weight量化节点的引入的，接下来我们可以看一下FX是如何实现对activation节点的插入。尽管FX的图表示支持在一张已创建好的图的任意节点前后进行插入和删除操作，目前FX却没有采用这种方式来进行activation量化节点的插入。相反地，FX采取的策略是：直接创建一张空图，然后按照原网络对应图表示的节点的拓扑顺序，依次向空图中拷贝原图中的节点，并根据该节点的qconfig是否包含了activation量化配置来决定是否在拷贝节点后插入相应的量化节点。

了解了FX插入activation量化节点的方式后，我们来具体看一下它的代码实现。

**4.1 量化模式匹配（pattern match）**

这是FX量化节点插入中很有特色的一个部分。在进行量化节点插入的过程中，有时候我们会希望根据特定节点之间的连接模式来决定如何插入activation量化。为此，FX中引入了模式匹配机制。这些模式匹配不仅在量化中，在fuse中也被使用，此处我们只介绍量化中的使用。

一个典型的activation量化插入和节点连接模式有关的例子是elementwise add。下图中展示了一个residual block，其中跨层的x和，最终通过

```python
@register_quant_pattern(torch.nn.intrinsic.qat.ConvBn1d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvBn2d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvBn3d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvBnReLU1d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvBnReLU2d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvBnReLU3d)
```

```python
@register_quant_pattern((torch.nn.ReLU, operator.add))
@register_quant_pattern((torch.nn.ReLU, operator.mul))
@register_quant_pattern((torch.nn.ReLU, torch.add))
@register_quant_pattern((torch.nn.ReLU, torch.mul))
@register_quant_pattern((torch.nn.functional.relu, operator.add))
@register_quant_pattern((torch.nn.functional.relu, operator.mul))
@register_quant_pattern((torch.nn.functional.relu, torch.add))
@register_quant_pattern((torch.nn.functional.relu, torch.mul))
```


**4.2 创建空图**

以下为FX中创建空图的代码片段，可以看到，在创建空图的同时，还有其他一些变量也随之创建。

```python
self.activation_post_process_map = dict()
env: Dict[Any, Any] = {}
observed_graph = Graph()
observed_node_names_set: Set[str] = set()
```

**4.3 根据节点类型和QConfig插入量化节点**

这是activation量化节点插入环节中最关键也最繁琐的一步，实际上在最新的Pytorch实现中，FX开发者已经对这里的逻辑进行了重构，代码更加简洁清晰。不过此处我们仍然按照v1.8.0版本的代码为准，进行解读。




**FAQ汇总**：

1. 在FX中如何自定义模块融合策略？
2. FX中目前已知的一些缺陷和问题？
   
   1.1 Bias Correction （BC）的实现方式：目前的BC是一个未被公开的API，但用户仍然可以通过调用XX获得BC。

3. 