### 符号跟踪（Symbolic Trace）

1. **图的表示**
   
FX中的核心数据结构（包括量化相关的部分）目前基本分布在torch/fx和torch/quantization两个文件夹下：
   - Node：在torch/fx/node.py
   - Graph：在
   - GraphModule：在
   - FakeQuantize：在torch/quantization/fake_quantize
   - Observer：
   - QConfig：QConfig中定义了weight和activation的量化方式

1.1 图上的操作



1. **Trace过程**

