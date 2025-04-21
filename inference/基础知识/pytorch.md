#Tensor

## 什么是稀疏矩阵？
稀疏矩阵是指其中绝大多数元素为零的矩阵。这类矩阵在存储和计算时可通过优化方式显著提高效率，常见于科学计算、机器学习等领域。
- 核心特点
1. 高零元素比例：非零元素数量远少于总元素数，通常无严格阈值，但需满足存储或计算优化的条件。
2. 高效存储：采用特殊结构（如COO、CSR、CSC）仅记录非零元素的位置和值，节省空间。
3. 计算优化：算法跳过零元素操作，减少不必要的计算量
- 存储格式示例：
1. COO（坐标格式）：存储三元组（行索引、列索引、值）。
2. CSR（压缩稀疏行）：压缩行指针，适合行操作。
3. CSC（压缩稀疏列）：类似CSR，但针对列优化。
- 应用场景
1. 自然语言处理：词袋模型中的文档-词矩阵。
2. 社交网络分析：用户关系图（邻接矩阵）。
3. 数值模拟：微分方程离散化后生成的矩阵。	
- 示例对比
1. 密集矩阵：1000×1000矩阵需存储1,000,000个元素。
2. 稀疏矩阵（100个非零）：COO格式仅需存储100×3=300个数据单元（通过行索引、列索引、值存储），空间节省显著。
## 例子
- [x] ![image](https://github.com/wangyd1988/AI-learning/blob/main/images/稀疏矩阵.png)
```
import torch
i = torch.tensor([[0, 1, 2],    # 行索引
                       [0, 1, 2]])   # 列索引
v = torch.tensor([1, 2, 3])     # 每个位置上的值
a = torch.sparse_coo_tensor(i, v, (4, 4))  # 构造 4x4 的稀疏矩阵
```
## 讲解
- . i: 表示稀疏矩阵中非零元素的位置（索引）
   i.shape = (2, 3) 表示 3 个坐标，每个坐标有两个值（row, col）：
```
i = [[0, 1, 2],   → 行坐标
     [0, 1, 2]]   → 列坐标
```	
也就是非零值的位置分别是：(0, 0)，(1, 1)，(2, 2)
- v: 对应于每个位置上的值，v = [1, 2, 3]
代表
1. 在位置 (0, 0) 放 1
2. 在位置 (1, 1) 放 2
3. 在位置 (2, 2) 放 3
-  a torch.sparse_coo_tensor(i, v, (4, 4))：
表示你构造了一个大小为 4×4 的稀疏张量，只在上面指定的位置存放值，其余地方默认是 0。
- a.to_dense() 用稠密形式（dense）打印出来是：
```
tensor([[1, 0, 0, 0],
          [0, 2, 0, 0],
          [0, 0, 3, 0],
          [0, 0, 0, 0]])
```
## 乘法运算
- 哈达玛积（element wise,对应元素相乘）
c = a * b等
## 举证运算
- 二维矩阵乘法运算操作包括torch.mm(),torch.matmul(),@  例如**(m,n)x(n,p)**
- 高维Tensor相乘(dim>2),定义其矩阵乘法仅在最后的两个维度上，**要求前面的维度必须保持一致**，就像举证的索引一样并且运算操作只有torch.matmul(),例如：
```
a = torch.ones(1,2,3,4)
b = torch.ones(1,2,4,3)
print(a.matmul(b));  print(torch.matmul(a,b))
```
## 幂运算
e^n  
print(torch.exe(a)) <=> b=a.exep_()
## 对数运算
- 对数运算是指数运算的“逆运算”，它回答的是这样一个问题：
1. 2^3=8 那么log2 8=？ => 2 的几次方等于 8？答案是 3。
2.torch.log2(a)=log2(a)栗子
```
import torch
a = torch.tensor([1, 2, 4, 8])
result = torch.log2(a)
print(result)
```
输出
```
tensor([0., 1., 2., 3.])
```

## 取整/取余运算
- .floor()向下取整数
- .ceil()向上取整数
- .round()四舍五入
- .trunc()裁剪，只取整数
- .frac()只取小数部分
- %取余
## 比较运算
- torch.eq(input,other,out=None)#按成员进行等式操作，相同返回True
- torch.equal(tensor1,tensor2) #如果tensor1和tensor2有相同的size和elements,则为true
- 高阶操作
1. torch.sort(input,dim=None,descending=False,out=None)#对目标input进行排序
2. torch.topk(input,k,dim=None,largest=True,sorted=True,out=None)#沿着指定维度返回最大K个数值及其索引值
3. torch.kthvalue(input,k,dim=None,out=None)#沿着指定维度返回第K个最小值及及其索引值
其中1,2主要在训练时loss的hard值

# Pytorch广播机制
- 广播机制：张量参数可以自动扩展为相同大小
- 必须满足至少两个条件：
1. 每个张量至少有一个维度
2. 满足右对齐
3. torch.rand(2,1,1)+torch.rand(3)