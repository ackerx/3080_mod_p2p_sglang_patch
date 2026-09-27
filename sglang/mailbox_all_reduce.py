# mailbox_all_reduce.py — 信箱 push allreduce communicator（BAR1 窗口 P2P 的 sglang 接入）
# 原理: 每卡一块 64MB 信箱(驱动加载后首批分配,
# 落 BAR1 窗口2), allreduce = 各 rank 把输入写进对端信箱(PCIe posted write 快方向
# ~10.28GB/s), 对端本地读并相加。单调 round 锁步协议, kernel 参数全指针,
# decode cuda graph capture/重放兼容(mailbox_graph 探针 500 次重放 PASS)。
# 与 CustomAllreduce 的差异: 无需 register_graph_buffers(信箱地址固定),
# 从根上绕开"129 个 graph buffer 动态分配必越窗"的墙。
# 启用: SGLANG_MAILBOX_AR=1 (dispatch_custom_allreduce 分支)
import contextlib
import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_MB_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mailbox_ar_ext.cu")
_MBX_BYTES = 64 << 20          # 信箱总尺寸(2MB 对齐, 数据区 48MB=两半×24MB)
_HALF_BYTES = 24 << 20         # 单次 allreduce 上限
_SHIM_BYTES = 64 << 20         # 落窗垫片(仅首轮): 占住窗口下沿的空洞, 把信箱顶进窗口

_ops = None


def _load_ops():
    global _ops
    if _ops is None:
        from torch.utils.cpp_extension import load
        # CUDA 13 拆包布局: cusparse.h 等头在 pip 的 nvidia/cu13/include
        nvidia_inc = os.path.join(os.path.dirname(torch.__file__),
                                  "..", "nvidia", "cu13", "include")
        extra = ["-O2", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
                 "-arch=sm_86"]   # 必须显式: wheel nvcc 13.3 默认 PTX 版本 > 驱动 JIT 支持
        if os.path.isfile(os.path.join(nvidia_inc, "cusparse.h")):
            extra.append("-I" + os.path.abspath(nvidia_inc))
        _ops = load(
            name="sglang_mailbox_ar_ext",
            sources=[_MB_SRC],
            extra_cuda_cflags=extra,
            verbose=bool(os.environ.get("SGLANG_MAILBOX_AR_VERBOSE")),
        )
    return _ops


try:
    from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
        is_weak_contiguous,
    )
except Exception:  # pragma: no cover
    def is_weak_contiguous(inp: torch.Tensor) -> bool:
        return inp.is_contiguous()


_DTYPE_IDS = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}


class MailboxAR:
    def __init__(self, group, device, max_size=_HALF_BYTES) -> None:
        self.disabled = True
        self.group = group
        self.max_size = max_size
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        if self.world_size != 2:
            logger.info("[MailboxAR] 仅支持 TP=2, 退回 NCCL")
            return
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        self.device = device
        torch.cuda.set_device(device)
        self.ops = _load_ops()

        # ---- 信箱分配: 必须落在 BAR1 窗口2(驱动加载后首批分配) ----
        # 尺寸重试 + IPC 打开成败作为落窗判据(同 mailbox_graph 探针逻辑)
        # 锁步纪律: 两个 all_gather_object 每轮必须全员参与, 且绝不能放进 try ——
        # 一侧失败就跳过 gather 的话, 两侧集合通信错代配对, 会开到对方已释放的
        # 旧一代缓冲(实测表现为一侧 ready 一侧永远"未落窗", 最后卡死在 gather 上)
        # 每轮协议: gather1 交换句柄 → 各自试开对端缓冲 → gather2 交换开结果;
        #   双方都开成功 = 共同落窗(两侧缓冲互为对方成功打开的那块, 天然一致);
        #   对端开我的缓冲成功 = 我的缓冲已证明在窗内, 保留复用; 否则丢弃重分。
        self.mbx = None
        self.peer_mbx = 0
        ok = False
        ok_attempt = -1
        shim = None
        for attempt in range(16):
            payload = None
            try:
                if self.mbx is None:
                    if attempt == 0:
                        # 垫片预推: 驱动把新分配放进窗口下沿的空洞(实测 ~70-88MB、
                        # 起点 ~222MB), 装得下就永远轮不到窗口区。先保留 64MB 垫片
                        # 占掉空洞大半, 信箱的 64MB 请求装不下即被顶进窗口, 首轮命中。
                        # 信箱定位后垫片立刻释放(不影响已定位置); 若空洞过小(<64MB)
                        # 垫片会短暂占用窗口使本轮作废, 下一轮起自动退化为常规
                        # +2MB 游走(空洞已归还, 行为与无垫片时完全一致)。
                        shim = torch.empty(_SHIM_BYTES, dtype=torch.uint8, device=device)
                    self.mbx = torch.empty(_MBX_BYTES + attempt * (2 << 20),
                                           dtype=torch.uint8, device=device)
                    if shim is not None:
                        shim = None                # 信箱位置已定, 垫片使命结束
                        torch.cuda.empty_cache()   # 归还驱动, 保持空闲链表干净
                payload = self.ops.ipc_get_handle(self.mbx.data_ptr())
            except Exception as e:
                logger.warning("[MailboxAR] attempt %d failed: %s", attempt, e)
                self.mbx = None
                payload = None
                shim = None
                torch.cuda.empty_cache()
            handles = [None] * self.world_size
            dist.all_gather_object(handles, payload, group=group)
            opened = False
            if handles[1 - self.rank] is not None:
                try:
                    self.peer_mbx = self.ops.ipc_open(handles[1 - self.rank])
                    opened = True
                except Exception as e:
                    logger.warning("[MailboxAR] attempt %d failed: %s (信箱未落窗?)",
                                   attempt, e)
                    torch.cuda.empty_cache()
            results = [False] * self.world_size
            dist.all_gather_object(results, opened, group=group)
            if all(results):
                ok = True
                ok_attempt = attempt
                break
            if not results[1 - self.rank]:
                # 对端没能打开我的缓冲(或本轮没测) → 我的缓冲未证明在窗内, 丢弃重分
                self.mbx = None
                torch.cuda.empty_cache()
        # 成功/回退必须两侧一致 —— 否则一侧走信箱一侧走 NCCL, 集合通信失配
        flags = [False] * self.world_size
        dist.all_gather_object(flags, ok, group=group)
        if not ok or not all(flags):
            if ok:
                self.mbx = None
                torch.cuda.empty_cache()
            logger.error("[MailboxAR] 信箱落窗失败(16 轮重试耗尽) — 本次回退 NCCL;"
                         "落窗与启动期分配布局有关, 重跑 start 通常可恢复")
            self.disabled = True
            return

        self.st = torch.empty(8, dtype=torch.int32, device=device)
        self.ops.state_reset(self.st.data_ptr())
        # 分段 AR(#8): flag 槽清 0(seq 从 1 起, 防垃圾初值误通过), host 侧 seq 计数
        self.ops.mailbox_seg_reset(self.mbx.data_ptr())
        self._seg_seq = 0
        self._seg_stream = None
        # 槽可容纳的段字节数(须与 .cu 的 MBX_SLOT_BYTES-512 一致)
        self.seg_slot_capacity = 5505024 - 512
        self.disabled = False
        logger.info("[MailboxAR] ready rank=%d attempt=%d mbx=%#x peer=%#x",
                    self.rank, ok_attempt, self.mbx.data_ptr(), self.peer_mbx)

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if inp.dtype not in _DTYPE_IDS:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        return inp_size <= self.max_size

    def custom_all_reduce(self, input: torch.Tensor):
        if self.disabled or not self.should_custom_ar(input):
            return None
        out = torch.empty_like(input)
        self.ops.mailbox_allreduce(
            input.data_ptr(), out.data_ptr(), input.numel(),
            _DTYPE_IDS[input.dtype],
            self.mbx.data_ptr(), self.peer_mbx, self.st.data_ptr(),
        )
        return out

    def capture(self):
        # 信箱地址固定, graph 内 kernel 直接记录指针, 无需注册/特殊处理
        return contextlib.nullcontext()

    # ---- 分段 AR(#8, CE push 版): 段提交在调用方副流(与 GEMM 重叠), 段等待在主流 ----
    def get_seg_stream(self):
        """段提交副流(惰性): CE push 与主流 GEMM 硬件并行(微基准实测净赚 ~1ms/AR)。"""
        if self._seg_stream is None:
            self._seg_stream = torch.cuda.Stream()
        return self._seg_stream

    def submit_seg(self, input: torch.Tensor, stream) -> int:
        """把一段输入 CE push 到对端槽位并写 flag(在 stream 上排队, 立即返回)。
        seq 由本端 host 单调递增维护, 与 fb 路径的 round/eflag 完全独立。"""
        assert not self.disabled
        self._seg_seq += 1
        self.ops.mailbox_submit_seg(
            input.data_ptr(), input.numel() * input.element_size(), self._seg_seq,
            self.peer_mbx, self.st.data_ptr(), stream.cuda_stream,
        )
        return self._seg_seq

    def wait_seg(self, seq: int, local_seg: torch.Tensor, out_seg: torch.Tensor) -> None:
        """主流上等对端第 seq 段 flag 到达并归约: out = local + 对端段。"""
        self.ops.mailbox_wait_seg(
            seq, local_seg.data_ptr(), out_seg.data_ptr(),
            local_seg.numel() * local_seg.element_size(),
            _DTYPE_IDS[local_seg.dtype],
            self.mbx.data_ptr(), self.st.data_ptr(), 0,   # 0 = 当前流(主流)
        )

    def seg_err(self) -> int:
        """分段路径错误码(0=正常), st[5]: 4=段 flag 等待超时。"""
        return int(self.ops.state_get(self.st.data_ptr(), 5))

    def register_graph_buffers(self):
        pass

    def close(self):
        self.disabled = True

    def __del__(self):
        self.close()
