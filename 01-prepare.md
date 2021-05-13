### Prepare流程

***

> 工欲善其事必先利其器 -- 《论语·卫灵公》

***

熟悉QAT流程的读者肯定知道，如果希望对一个模型进行QAT量化，一般需要对网络中的权重（weight）和激活值（activation）进行模拟量化，并在训练中借助模拟量化节点反向传播，对网络进行训练。

一个普通的nn.Module是如何转换得到插入模拟量化节点的模型的呢？在FX中，这个转换过程大体分为四个步骤：

（1）符号追踪（symbolic trace）:获得原始nn.Module网络的图表示

（2）模块融合（fuse）: 此步骤将满足模式的相邻模块融合成nn.Sequential。例如，将相邻的nn.Conv2d和nn.BatchNorm2d打包到一个nn.Sequential内，构成一个FusedModule，此后这个FusedModule将在prepare环节被替换成一个融合的模块。例如nn.Conv2d+nn.BatchNorm2d -> nn.intrinsic.qat.ConvBN2d。

（3）模块交换（swap）：此步骤是为了实现weight的模拟量化。由于nn.Conv2d中的weight是不带模拟量化（Fake Quantize）功能的，为了实现QAT中的模拟量化，需要把所有普通模块替换成可以进行模拟量化的QAT module，而这些module一般都定义在torch.nn.qat当中。torch官方已经对这些模拟量化module的forward和backward进行了实现，可以在训练过程中实现反向传播。例如nn.Conv2d -> nn.qat.intrinsic.Conv2d。

（4）activation量化节点插入（insert observer）：此步骤是为了实现对activation的模拟量化（或者在PTQ中插入观察节点，实现对激活值数据分布的观察）。

以上即为一个普通的nn.Module转换为一个可进行QAT的graph module的大致流程。本文后续内容将深入每个环节对其中包含的诸多操作进行拆解，条分缕析。

***

0. 预备内容
   
FX中的核心数据结构（包括量化相关的部分）目前基本分布在torch/fx和torch/quantization两个文件夹下：
   - Node：在torch/fx/node.py
   - Graph：在
   - GraphModule：在
   - FakeQuantize：在torch/quantization/fake_quantize
   - Observer：
   - QConfig：QConfig中定义了weight和activation的量化方式

1. QAT Prepare

```python
prepared = quantizer.prepare(graph_module, 
                             qconfig_dict,
                             tracer.node_name_to_scope,
                             prepare_custom_config_dict=prepare_custom_config_dict,
                             is_standalone_module=is_standalone_module
                            )
```

此处即prepare的入口。


3. 模块交换（swap）

3.1 准备工作 —— 传播量化配置（propagate）
    
在实际量化中，网络中各个层的量化配置（权重位宽、激活值位宽等）未必完全相同。为此FX允许用户定义

3.2 模块替换

在传播完成量化配置后，可以对每个模块


4. activation量化节点插入

4.1 创建空图

4.1 根据节点类型和QConfig插入量化节点