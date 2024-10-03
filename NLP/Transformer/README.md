# 使用nn.Transformer实现一个简单的Copy任务

**任务描述：**让Transformer预测输入。例如，输入为`[0, 3, 4, 6, 7, 1, 2, 2]`，则期望的输出为`[0, 3, 4, 6, 7, 1]`。

这里是一个翻译任务中transformer的输入和输出。transformer的输入包含两部分：

- inputs: 原句子对应的tokens，且是完整句子。一般0表示句子开始(<bos>)，1表示句子结束(<eos>)，2为填充(<pad>)。填充的目的是为了让不同长度的句子变为同一个长度，这样才可以组成一个batch。在代码中，该变量一般取名src。
- outputs(shifted right)：上一个阶段的输出。虽然名字叫outputs，但是它是输入。最开始为0（<bos>），然后本次预测出“我”后，下次调用Transformer的该输入就变成<bos> 我。在代码中，该变量一般取名tgt。
