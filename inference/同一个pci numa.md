# 启动mst 工具，并查看网卡信息。
```
这里可以看到，ConnectX5双口网卡的第一个接口的信息为：
mst地址为/dev/mst/mt4121_pciconf0
PCI地址为06:00.0
RDMA设备名称为mlx5_0
网络设备名称为enp6s0f0
位于NUMA-1


-[0000:00]: 这是PCIe Host Bridge，代表CPU的PCIe根联合体。通常，一个多CPU系统会有多个Host Bridge（例如 [0000:00] 和 [0000:80]）。

```
# 使用lshw命令
```
sudo lshw -class bridge -class display -class network
这个命令会列出所有PCIe桥、显示控制器（GPU）和网络控制器的详细信息，包括它们的bus info（总线信息），您可以根据总线信息来追溯它们的连接关系。
```



- [x] ![image](https://github.com/wangyd1988/AI-learning/blob/main/images/rdma-gpu-cpu-1.png)
- [x] ![image](https://github.com/wangyd1988/AI-learning/blob/main/images/rdma-gpu-cpu-2.png)