> https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html
Transformer视频&文件
> https://youtu.be/ugWDIIOHtPA
> http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2019/Lecture/Transformer%20%28v5%29.pdf
# RNN
RNN很难平行化，
![image](https://camo.githubusercontent.com/5072f18534482a4e93dc7b71889f49e8be273c373b3ee5219612e54e75b0b9c9/68747470733a2f2f6c65656d656e672e74772f696d616765732f7472616e73666f726d65722f726e6e2d76732d73656c662d6174746e2d6c617965722e6a7067)
在上图中要计算出$$b^4$$的值，RNN需要经过4个时间点依次看过[a1,a2,a3,a4]以后才能在序列中获取最后一个输出$$b^4$$的值。而子注意层（Self-Attention Layer）可以利用矩阵覆盖在RNN的一个时间点内回传所有$$b^i$$,且每个bi都包含了整个输入序列的信息
# transformer:Attention is all you need
- 利用self-Attention Layer来替代RNN,但可以做到**平行化**，不需要计算前N个内容就可以计算出N+1个内容

## transformation也即Q,K,V
transformation:改观，变化，转变
$$q^i=W^q\a^i$$


### 结构
![image](https://p1-jj.byteimg.com/tos-cn-i-t2oaga2asx/gold-user-assets/2018/9/17/165e5814fae0765f~tplv-t2oaga2asx-jj-mark:3024:0:0:0:q75.awebp)
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

[![点击播放 Transformer 演示视频](https://leemeng.tw/images/transformer/transformer-cover.png)](https://leemeng.tw/images/transformer/transformer-nmt-encode-decode.mp4)
- 以Transformer 实作的NMT 系统基本上可以分为6 个步骤：
1. Encoder 为输入序列里的每个词汇产生初始的repr. （即词向量），以空圈表示
2. 利用self-Attention机制将序列中的所有词汇，的语义资讯各自汇总成每个词汇的repr.，以实圈表示
3. Encoder重复N次注意力机制，让每个词汇的repr彼此持续修正来完整纳入上下文
4. Decoder在生成每个文字时也应用了自注意力机制，关注自己之前生成的元素，将语义纳入之后的元素
5. 在自注意力机制后，Decoder 接着利用注意力机制关注Encoder 的所有输出并将其资讯纳入当前生成元素的repr.
6. Decoder 重复步骤4, 5 以让当前元素完整包含整体语义

## 魔搭讲解transfomer
![image](https://p1-jj.byteimg.com/tos-cn-i-t2oaga2asx/gold-user-assets/2018/9/17/165e5814fae0765f~tplv-t2oaga2asx-jj-mark:3024:0:0:0:q75.awebp)

![image](https://github.com/wangyd1988/AI-learning/blob/main/images/transformer-modelscope.png)