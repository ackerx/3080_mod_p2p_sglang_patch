# mailbox_sweep.py — 信箱 vs NCCL 按消息尺寸扫描（eager, 2 进程, TP=2）
# 用法: 魔改驱动 + BAR1 窗口空闲(先停 GPU 服务); python mailbox_sweep.py
# 输出: 各尺寸下两路径的延迟/有效带宽(algbw, n=2 时 busbw=algbw)与比值
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

PORT = 29517
SIZES = [64 << 10, 256 << 10, 1 << 20, 4 << 20, 8 << 20, 10240 << 10, 16 << 20, 24575 << 10]
ITERS = 50


def run(rank):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{PORT}",
                            rank=rank, world_size=2)
    import sglang.srt.distributed.device_communicators.mailbox_all_reduce as m
    # 垫片预推是为 sglang 服务的内存布局(权重占满、222MB 处有空洞)设计的;
    # 全新空进程布局不同, 垫片反而扰乱落窗。这里关闭垫片, 复刻
    # test_mailbox_torch.py 已验证成功的条件(靠 +2MB 游走落窗)。
    m._SHIM_BYTES = 0
    from sglang.srt.distributed.device_communicators.mailbox_all_reduce import MailboxAR
    ar = MailboxAR(None, rank)
    assert not ar.disabled, "MailboxAR 落窗失败, 无法扫描"
    if rank == 0:
        print("== mailbox ready, sweep start ==", flush=True)

    for sz in SIZES:
        n = sz // 2
        t = torch.ones(n, dtype=torch.bfloat16, device=f"cuda:{rank}")

        # --- NCCL ---
        for _ in range(10):
            dist.all_reduce(t)
        torch.cuda.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            dist.all_reduce(t)
        torch.cuda.synchronize(); dist.barrier()
        nccl_us = (time.perf_counter() - t0) / ITERS * 1e6

        # --- mailbox ---
        # NCCL 的 all_reduce 是原地累加, 上面 60 轮已把 t 溢出成 inf, 重置后再测
        t = torch.ones(n, dtype=torch.bfloat16, device=f"cuda:{rank}")
        for _ in range(10):
            out = ar.custom_all_reduce(t)
        torch.cuda.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            out = ar.custom_all_reduce(t)
        torch.cuda.synchronize(); dist.barrier()
        mbx_us = (time.perf_counter() - t0) / ITERS * 1e6

        ok = bool(torch.all(out.float() == 2.0).item())
        if rank == 0:
            w_nccl = sz / (nccl_us / 1e6) / 1e9
            w_mbx = sz / (mbx_us / 1e6) / 1e9
            print(f"{sz>>10:7d}KB  NCCL {nccl_us:9.1f}us {w_nccl:6.2f}GB/s | "
                  f"MBX {mbx_us:9.1f}us {w_mbx:6.2f}GB/s | "
                  f"MBX快 {nccl_us/mbx_us:5.2f}x  ok={ok}", flush=True)

    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(run, nprocs=2, join=True)
    print("SWEEP-DONE")
