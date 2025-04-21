> https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html
# RNN
RNN很难平行化，
# transformer:Attention is all you need
- 利用self-Attention Layer来替代RNN,但可以做到**平行化**，不需要计算前N个内容就可以计算出N+1个内容

## transformation也即Q,K,V
transformation:改观，变化，转变
$$q^i=W^q\a^i$$
### 概念
- 自注意层（Self-Attention Layer）跟RNN 一样，输入是一个序列，输出一个序列。但是该层可以平行计算，且输出序列中的每个向量都已经看了整个序列的资讯。
- 自注意层将输入序列I里头的每个位置的向量i透过3 个线性转换分别变成3 个向量：q、k和v，并将每个位置的q拿去跟序列中其他位置的k做匹配，算出匹配程度后利用softmax 层取得介于0到1之间的**权重值**，并以此权重跟每个位置的v作加权平均，最后取得该位置的输出向量o。全部位置的输出向量可以同时平行计算，最后输出序列O。
- 计算匹配程度（注意）的方法不只一种，只要能吃进2 个向量并吐出一个数值即可。但在Transformer 论文原文是将2向量做dot product 算匹配程度。
- 可以透过大量矩阵运算以及GPU 将概念2 提到的注意力机制的计算全部平行化，加快训练效率（也是本文实作的重点）。
- 多头注意力机制（Multi-head Attention）是将输入序列中的每个位置的q、k和v切割成多个$$q^i、k^i和v^i$$再分别各自进行注意力机制。各自处理完以后把所有结果串接并视情况**降维**。这样的好处是能让各个head 各司其职，学会关注序列中不同位置在不同representaton spaces 的资讯。
- 自注意力机制这样的计算的好处是「天涯若比邻」：序列中每个位置都可以在O(1) 的距离内关注任一其他位置的资讯，运算效率较双向RNN 优秀。
- 自注意层可以取代Seq2Seq 模型里头以RNN 为基础的Encoder / Decoder，而实际上全部替换掉后就（大致上）是Transformer。
- 自注意力机制预设没有「先后顺序」的概念，而这也是为何其可以快速平行运算的原因。在进行如机器翻译等序列生成任务时，我们需要**额外加入位置编码（Positioning Encoding）**来加入顺序资讯。而在Transformer 原论文中此值为**手设**而非训练出来的模型权重。
- Transformer 是一个Seq2Seq 模型，自然包含了Encoder / Decoder，而Encoder及Decoder 可以包含多层结构相同的blocks，里头每层都会有multi-head attention 以及Feed Forward Network。
- 在每个Encoder / Decoder block 里头，我们还会使用残差连结（Residual Connection）以及Layer Normalization。这些能帮助模型稳定训练。
- Decoder 在关注Encoder 输出时会需要遮罩（mask）来避免看到未来资讯。我们后面会看到，事实上还会需要其他遮罩。
## transformer快速理解

<video src="https://leemeng.tw/images/transformer/transformer-nmt-encode-decode.mp4" controls width="600">
  您的浏览器不支持 HTML5 视频标签。
</video>
## transformer理解视频

