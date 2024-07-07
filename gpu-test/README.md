# DataLoader中num_workers的选择

## 参数详解：

1. 每次dataloader加载数据时：dataloader一次性创建num_worker个worker，（也可以说dataloader一次性创建num_worker个工作进程，worker也是普通的工作进程），并用batch_sampler将指定batch分配给指定worker，worker将它负责的batch加载进RAM。
2. 然后，dataloader从RAM中找本轮迭代要用的batch，如果找到了，就使用。如果没找到，就要num_worker个worker继续加载batch到内存，直到dataloader在RAM中找到目标batch。一般情况下都是能找到的，因为batch_sampler指定batch时当然优先指定本轮要用的batch。
3. num_worker设置得大，好处是寻batch速度快，因为下一轮迭代的batch很可能在上一轮/上上一轮...迭代时已经加载好了。坏处是内存开销大，也加重了CPU负担（worker加载数据到RAM的进程是CPU复制的嘛）。num_workers的经验设置值是自己电脑/服务器的CPU核心数，如果CPU很强、RAM也很充足，就可以设置得更大些。
4. 如果num_worker设为0，意味着每一轮迭代时，dataloader不再有自主加载数据到RAM这一步骤（因为没有worker了），而是在RAM中找batch，找不到时再加载相应的batch。缺点当然是速度更慢。

## 设置大小建议：

1. num_workers=0表示只有主进程去加载batch数据，这个可能会是一个瓶颈。
2. num_workers = 1表示只有一个worker进程用来加载batch数据，而主进程是不参与数据加载的。这样速度也会很慢。
3. num_workers>0 表示只有指定数量的worker进程去加载数据，主进程不参与。增加num_works也同时会增加cpu内存的消耗。所以num_workers的值依赖于 batch size和机器性能。
4. 一般开始是将num_workers设置为等于计算机上的CPU数量
5. 最好的办法是缓慢增加num_workers，直到训练速度不再提高，就停止增加num_workers的值。
   

## 参数测试

运行一遍gpu-test.py即可
