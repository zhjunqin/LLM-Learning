# MPI 

这里介绍各种 MPI 集合通信 Collective Operations。


## Overview

Overview of collective operations. 
- (a) Broadcast. 
- (b) All-Gather.
- (c) Scatter.
- (d) All-to-All.
- (e) Reduce.
- (f) All-Reduce.
- (g) Reduce-Scatter.

![](./assets/10.1007-s11390-023-2894-6-Figure2.jpg)


## NCCL Collective Operations

### ALLReduce

AllReduce 操作在不同设备之间对数据进行归约操作（例如求和、最小值、最大值），并将结果存储在每个排名的接收缓冲区中。

在 k 个 Rank 之间的求和 AllReduce 操作中，每个 Rank 将提供一个包含 N 个值的数组 $in$，并在数组 $out$ 中接收相同的结果，其中 $out[i] = in0[i] + in1[i] + ... + in(k-1)[i]$。

![](./assets/nccl_allreduce.png)

### Broadcast

Broadcast 操作将一个包含 N 个元素的缓冲区从 root Rank 复制到所有 Rank。

注意：root 参数是 Rank 之一，而不是设备编号，因此受到不同的 Rank 到设备映射的影响。

![](./assets/nccl_broadcast.png)

### Reduce

Reduce 操作执行与 AllReduce 相同的操作，但只将结果存储在指定 root Rank 的接收缓冲区中。

![](./assets/nccl_reduce.png)

注意：root 参数是 Rank 之一（而不是设备编号），因此受到不同的 Rank 到设备映射的影响。

注意：Reduce 操作后跟 Broadcast 操作等效于 AllReduce 操作。


### AllGather

AllGather 操作将来自 k 个 Rank 的 N 个值收集到一个大小为 k*N 的输出缓冲区，并将该结果分发给所有 Rank。

输出按 Rank 索引排序。因此，AllGather 操作受到不同的 Rank 到设备映射的影响。

![](./assets/nccl_allgather.png)

注意：AllReduce 操作等效于执行 ReduceScatter 操作，然后再执行 AllGather 操作。

### ReduceScatter

ReduceScatter 操作执行与 Reduce 相同的操作，但结果以相等大小的块在 Rank 之间分散，每个 Rank 根据其 Rank 索引获得一块数据。

由于 Rank 确定数据布局，因此 ReduceScatter 操作受到不同的 Rank 到设备映射的影响。

![](./assets/nccl_reducescatter.png)



## 参考文献
- https://jcst.ict.ac.cn/en/supplement/f82ea167-0ca9-46dc-9109-28b6ca0cf983
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html