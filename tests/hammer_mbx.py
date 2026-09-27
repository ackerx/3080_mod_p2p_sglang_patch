# hammer_mbx.py — MailboxAR 正确性锤子: 每轮变值 + 逐轮校验(专打 tail 竞态/L2-stale)
# 用法: 双卡机器, 魔改驱动 + BAR1 窗口空闲(先停 GPU 服务); python hammer_mbx.py
# 协议语义: 双 rank 同值输入 → out = 2×输入; 每轮输入随轮号变化, 同步后立即校验。
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

PORT = 29521


def run(rank):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{PORT}",
                            rank=rank, world_size=2)
    import sglang.srt.distributed.device_communicators.mailbox_all_reduce as m
    m._SHIM_BYTES = 0
    from sglang.srt.distributed.device_communicators.mailbox_all_reduce import MailboxAR
    ar = MailboxAR(None, rank)
    assert not ar.disabled, "MailboxAR 落窗失败"
    if rank == 0:
        print("mailbox ready, hammer start", flush=True)

    bad = 0
    for phase, (sz, iters) in enumerate([(10485760, 300), (65536, 200), (24575 << 10, 60)]):
        n = sz // 2
        mism = 0
        for it in range(iters):
            v = (it % 251) + 1.0
            t = torch.full((n,), v, dtype=torch.bfloat16, device=f"cuda:{rank}")
            out = ar.custom_all_reduce(t)
            ok = bool(torch.all(out.float() == 2.0 * v).item())
            if not ok:
                mism += 1
                if mism <= 3 and rank == 0:
                    d = (out.float() - 2 * v).abs().max().item()
                    idx = (out.float() - 2 * v).abs().argmax().item()
                    print(f"  MISMATCH phase{phase} iter{it} max|dev|={d} @idx={idx}", flush=True)
            dist.barrier()
        bad += mism
        if rank == 0:
            print(f"phase{phase} {sz>>10}KB x{iters}: mismatch={mism}", flush=True)
    if rank == 0:
        print(f"HAMMER-RESULT bad={bad} {'PASS' if bad == 0 else 'FAIL'}", flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(run, nprocs=2, join=True)
