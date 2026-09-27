# sglang 接入：MailboxAR communicator

在 sglang（验证版本 **0.5.19**，TP=2）中把 TP 组的 custom allreduce 替换为
信箱 push allreduce。两个新文件 + 一处 3 行接线。

## 安装

```bash
SGC=$(python3 -c 'import sglang,os;print(os.path.dirname(sglang.__file__))')
cp mailbox_ar_ext.cu      $SGC/srt/distributed/device_communicators/
cp mailbox_all_reduce.py  $SGC/srt/distributed/device_communicators/
```

## 接线（3 行）

编辑 `$SGC/srt/distributed/device_communicators/custom_all_reduce.py`，
在 `dispatch_custom_allreduce()` 函数体开头（其余分支判断之前）插入：

```python
    if _is_cuda and get_bool_env_var("SGLANG_MAILBOX_AR", default="false"):
        from .mailbox_all_reduce import MailboxAR
        logger.info("[AR] Using MailboxAR (BAR1 window P2P push mode)")
        return MailboxAR
```

`get_bool_env_var` 来自该文件已有的 import（sglang.srt.utils）。若你的
sglang 版本结构不同，接线原则就一条：让 TP group coordinator 的
`ca_comm` 实例化为 `MailboxAR` —— 它实现了
`should_custom_ar / custom_all_reduce / capture / register_graph_buffers`
与 CustomAllreduce 相同的接口面，decode CUDA graph 直接兼容（信箱地址
固定，graph 内 kernel 记录裸指针即可，无需 register_graph_buffers）。

## 启用

```bash
SGLANG_MAILBOX_AR=1 python -m sglang.launch_server ... --tp 2
```

首次启动会在用户态 JIT 编译 CUDA 扩展（约 1~2 分钟；缓存在
`~/.cache/torch_extensions/`，改了 .cu 后删除该目录强制重编）。

启动日志确认落窗：

```
[AR] Using MailboxAR (BAR1 window P2P push mode)
[MailboxAR] ready rank=0 attempt=0 mbx=0x... peer=0x...
```

`attempt>0` 表示信箱没能在首轮分配落进 BAR1 窗口（重试最多 16 轮）；
出现 `信箱落窗失败` 时重启一次服务通常可恢复（启动期显存分配顺序问题，
垫片预推逻辑见 `mailbox_all_reduce.py`）。

## 工作范围与门控

- `should_custom_ar`：dtype bf16/fp16/fp32、16B 对齐、≤24MB —— 超限自动
  回落 NCCL，与 CustomAllreduce 行为一致；
- decode 走 CUDA graph：信箱协议的 kernel 参数全为裸指针，capture/重放
  天然兼容，无需额外注册；
- `mailbox_ar_ext.cu` 中还包含一套**分段 AR 协议**（`submit_seg/wait_seg`，
  CE push 版，供 prefill 计算×通信重叠实验），默认无任何调用路径，不影响
  主流程；相关实验结论见 `docs/performance.md`。
