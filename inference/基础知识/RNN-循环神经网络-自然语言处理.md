独热编码
One-hot Encoding
/
wʌn hɒt ˈɛnkəʊdɪŋ
/
n. 一种将分类变量转换为二进制向量表示的方法

独热编码是一种常用的特征处理技术，主要应用于机器学习和深度学习模型中。它通过创建稀疏的二进制向量来表示每一个分类标签，在该向量中，只有与某个特定类别相对应的维度值为1，其余维度值均为0。例如，对于水果这一分类变量，包括苹果、香蕉和橙子，独热编码将苹果表示为 [1, 0, 0]，香蕉表示为 [0, 1, 0]，橙子则表示为 [0, 0, 1]。这样处理的好处在于避免了类别之间的顺序关系，使得模型可以更有效地进行学习和预测。

> https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html

# RNN循环神级网络基本步骤
- 处理输入文字
 1. 文本分词（Text Segmentation）
 2. 建立字典并将文本转成数字序列
 3. 序列的Zero Padding
 4. 将正解做One-hot Encoding
那么进行文字处理前，需要训练集数据读取过来，有了数据后，就可以开始处理数据了。

## - **文本分词**
文本分词（Text Segmentation）是一个将一连串文字切割成多个有意义的单位的步骤。这单位可以是
1. 一个中文汉字/ 英文字母（Character）
2. 一个中文词汇/ 英文单字（Word）
3. 一个中文句子/ 英文句子（Sentence）
- **建立字典并将文本转成数字序列**
例如 { '狐狸' : 0 , '被' : 1 , '陌生人' : 2 , '拍照' : 3 }       
对于大模型常用的Tokenizer来说，Tokenizer 顾名思义即是将一段文字转换成一系列的词汇（Tokens),并为其建立字典,如
[ '2017' , '养老保险' , '又' , '新增' , '两项' , '农村' , '老人' , '人人' , '可' , '申领' , '你' , '领到' , '了' , '吗' ]             
- **序列的Zero Padding**
> 'Zero Padding' 零填充 是指在处理数据（如图像、文本等）时，根据需要在数据的边缘填充零，以使输入数据的形状适应特定的模型要求。它在卷积神经网络（CNN）中常用，是保持处理特征的空间维度不变的常用技术。通过采用零填充，可以避免数据在边缘丢失，并在卷积操作后保持输出维度的一致性。

此技术特别适用于处理序列数据及图像数据，使得模型训练能够顺利进行，同时有助于提升模型的效果。
- **将正解做One-hot Encoding**
> 独热编码是一种常用的特征处理技术，主要应用于机器学习和深度学习模型中。它通过创建稀疏的二进制向量来表示每一个分类标签，在该向量中，只有与某个特定类别相对应的维度值为1，其余维度值均为0。例如，对于水果这一分类变量，包括苹果、香蕉和橙子，独热编码将苹果表示为 [1, 0, 0]，香蕉表示为 [0, 1, 0]，橙子则表示为 [0, 0, 1]。这样处理的好处在于避免了类别之间的顺序关系，使得模型可以更有效地进行学习和预测。
![image](https://leemeng.tw/images/nlp-kaggle-intro/one-encoding.jpg)

## 有记忆的循环神经网路
![image](https://leemeng.tw/images/nlp-kaggle-intro/rnn-static.png)
![image](https://leemeng.tw/images/nlp-kaggle-intro/rnn-animate.gif)
RNN 一次只读入并处理序列的「一个」元素
现在你可以想像为何RNN 非常适合拿来处理像是自然语言这种序列数据了。

就像你现在阅读这段话一样，你是由左到右逐字在大脑里处理我现在写的文字，同时不断地更新你脑中的记忆状态。

每当下个词汇映入眼中，你脑中的处理都会跟以下两者相关：

1. 前面所有已读的词汇
2. 目前脑中的记忆状态
当然，实际人脑的阅读机制更为复杂，但RNN 抓到这个处理精髓，利用内在回圈以及细胞内的「记忆状态」来处理序列资料。
![image](https://leemeng.tw/images/nlp-kaggle-intro/thought-catalog-196661-unsplash.jpg)

## 记忆力好的LSTM 细胞
长短期记忆(Long Short-Term Memory, 后简称LSTM)就是被设计来解决RNN的这个问题。
你可以把LSTM 想成是RNN中用来实现细胞A内部处理逻辑的一个特定方法：
![image](https://leemeng.tw/images/nlp-kaggle-intro/lstm-cell.png)
以抽象的层次来看，LSTM 就是实现RNN 中细胞A 逻辑的一个方式
基本上一个LSTM 细胞里头会有3 个闸门（Gates）来控制细胞在不同时间点的记忆状态：
1. Forget Gate：决定细胞是否要遗忘目前的记忆状态
2. Input Gate：决定目前输入有没有重要到值得处理
3. Output Gate：决定更新后的记忆状态有多少要输出
透过这些闸门控管机制，LSTM可以将很久以前的记忆状态储存下来，在需要的时候再次拿出来使用。值得一提的是，这些闸门的参数也都是神经网路自己训练出来的。
![image](https://leemeng.tw/images/nlp-kaggle-intro/lstm-cell-detailed.png)
LSTM 细胞顶端那条cell state 正代表着细胞记忆的转换过程
想像LSTM 细胞里头的记忆状态是一个包裹，上面那条直线就代表着一个输送带。LSTM 可以把任意时间点的记忆状态（包裹）放上该输送带，然后在未来的某个时间点将其原封不动地取下来使用。
![image](https://leemeng.tw/images/nlp-kaggle-intro/accumulation-conveyor-101.jpg)

## 词向量：将词汇表达成有意义的向量
事实上要让神经网路能够处理标题序列内的词汇，我们要将它们表示成向量（更精准地说，是张量：Tensor），而不是一个单纯数字。如果我们能做到这件事情，则RNN 就能用以下的方式读入我们的资料：
![image](https://leemeng.tw/images/nlp-kaggle-intro/rnn-process-vectors.gif)
所以现在的问题变成：
「要怎么将一个词汇表示成一个N 维向量？」
其中一个方法是我们随便决定一个N，然后为语料库里头的每一个词汇都指派一个随机生成的N 维向量。

假设我们现在有5 个词汇：
1. 野狼
2. 老虎
3. 狗
4. 猫
5. 喵咪
依照刚刚说的方法，我们可以设定N = 2，并为每个词汇随机分配一个2 维向量后将它们画在一个平面空间里头：
![image](https://leemeng.tw/images/nlp-kaggle-intro/2d-random-word-vector.jpg)
这些代表词汇的向量被称之为词向量，但是你可以想像这样的随机转换很没意义。
比方说上图，我们就无法理解：
1. 为何「狗」是跟「老虎」而不是跟同为犬科的「野狼」比较接近？
2. 为何「猫」的维度2 比「狗」高，但却比「野狼」低？
3. 维度2 的值的大小到底代表什么意义？
4.「喵咪」怎么会在那里？
这是因为我们只是将词汇随机地转换到2维空间，并没有让这些转换的结果（向量）反应出词汇本身的语意（Semantic）。
一个理想的转换应该是像底下这样：
![image](https://leemeng.tw/images/nlp-kaggle-intro/2d-good-word-vector.jpg)
在这个2 维空间里头，我们可以发现一个好的转换有2 个特性：
1. **距离有意**：「喵咪」与意思相近的词汇「猫」距离接近，而与较不相关的「狗」距离较远
2. **维度有意义**：看看（狗, 猫）与（野狼, 老虎）这两对组合，可以发现我们能将维度1 解释为猫科VS 犬科；维度2 解释为宠物与野生动物
如果我们能把语料库（Corpus）里头的每个词汇都表示成一个像是这样有意义的词向量，神经网路就能帮我们找到潜藏在大量词汇中的语义关系，并进一步改善NLP 任务的精准度。
好消息是，大部分的情况我们并不需要自己手动设定每个词汇的词向量。我们可以随机初始化所有词向量（如前述的随机转换），并利用平常训练神经网路的反向传播算法（Backpropagation），让神经网路自动学到一组适合当前NLP 任务的词向量（如上张图的理想状态）。
![image](https://leemeng.tw/images/nlp-kaggle-intro/backpropagation-example.gif)
反向传播让神经网路可以在训练过程中修正参数，持续减少预测错误的可能性
在NLP 里头，这种将一个词汇或句子转换成**一个实数词向量（Vectors of real numbers）的技术被称之为词嵌入（Word Embedding）。**

![image](https://leemeng.tw/images/nlp-kaggle-intro/siamese-network.jpg)
## 全连接层
全连接层顾名思义，代表该层的每个神经元（Neuron）都会跟前一层的所有神经元享有连结：

## Softmax 函式
Softmax 函式一般都会被用在整个神经网路的最后一层上面，比方说我们这次的全连接层。

Softmax 函式能将某层中的所有神经元里头的数字作正规化（Normalization）：将它们全部压缩到0 到1 之间的范围，并让它们的和等于1。
![image](https://leemeng.tw/images/nlp-kaggle-intro/softmax-and-fully-connectead.jpg)