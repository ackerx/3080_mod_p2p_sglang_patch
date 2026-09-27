# dbg8.py — #8 分段 AR(CE push) 协议正确性
# 用法: 双 rank 同时起 (CUDA_VISIBLE_DEVICES=0/1, argv[1]=rank)
# 覆盖: 偶/奇分段、单段退化、段与 fb 路径交替(st/eflag 互扰)、紧间隔
#       (submit 后立即 wait, push 在途)、段后 fb 回归
import os
import sys

import torch
import torch.distributed as dist

os.environ.setdefault("SGLANG_MAILBOX_AR", "1")
rank = int(sys.argv[1])
dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29588",
                        rank=rank, world_size=2)
# 跑法约定: 每 rank 进程以 CUDA_VISIBLE_DEVICES=<rank> 启动, 可见卡重编号为逻辑 0;
# 未设 CUDA_VISIBLE_DEVICES 时才直接用 rank 当逻辑号
dev = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else rank
torch.cuda.set_device(dev)
from sglang.srt.distributed.device_communicators.mailbox_all_reduce import MailboxAR

ar = MailboxAR(None, dev)
assert not ar.disabled, "MailboxAR 未落窗"
print(f"[r{rank}] MailboxAR ready", flush=True)

torch.manual_seed(7 + rank)
H = 5120
seg_stream = torch.cuda.Stream()
fails = 0


def expected_full(x):
    xc = x.cpu()
    parts = [torch.empty_like(xc) for _ in range(2)]
    dist.all_gather(parts, xc)
    return (parts[0] + parts[1]).to(x.device)


def check(tag, out, exp):
    global fails
    ok = torch.allclose(out.float(), exp.float(), rtol=2e-2, atol=2e-2)
    mx = (out.float() - exp.float()).abs().max().item()
    if not ok:
        fails += 1
    print(f"[r{rank}] {tag}: {'OK' if ok else 'FAIL'} max_err={mx:.4f}", flush=True)
    return ok


def seg_round(tag, S, k, fb_between, gap=True):
    x = torch.randn(S, H, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x)
    base = S // k
    bounds = [i * base for i in range(k)] + [S]
    seqs = []
    for i in range(k):
        seqs.append(ar.submit_seg(x[bounds[i]:bounds[i + 1]], seg_stream))
    exp = expected_full(x) if gap else None
    fb_pending = None
    if fb_between:
        fb_pending = ar.custom_all_reduce(x)
        assert fb_pending is not None
    for i in range(k):
        ar.wait_seg(seqs[i], x[bounds[i]:bounds[i + 1]],
                    out[bounds[i]:bounds[i + 1]])
    if not gap:
        exp = expected_full(x)      # wait 后才对账: wait 时 push 大概率在途
    torch.cuda.synchronize()
    e = ar.seg_err()
    assert e == 0, f"seg_err={e}"
    if fb_between:
        check(tag + "/fb", fb_pending, exp)
    check(tag, out, exp)


try:
    for rep in range(10):
        seg_round(f"even k=2 S=1024 #{rep}", 1024, 2, fb_between=True)
    for rep in range(5):
        seg_round(f"odd  k=3 S=1000 #{rep}", 1000, 3, fb_between=True)
    for rep in range(5):
        seg_round(f"one  k=1 S=256  #{rep}", 256, 1, fb_between=True)
    for rep in range(5):
        seg_round(f"tight k=2 S=1024 #{rep}", 1024, 2, fb_between=True, gap=False)
    for rep in range(10):           # 段路径停用后 fb 仍正常(回归)
        x = torch.randn(512, H, dtype=torch.bfloat16, device="cuda")
        exp = expected_full(x)
        out = ar.custom_all_reduce(x)
        assert out is not None
        check(f"fb-only #{rep}", out, exp)
finally:
    dist.barrier()
    total = [0, 0]
    dist.all_gather_object(total, fails)
    print(f"[r{rank}] RESULT fails={sum(total)}", flush=True)
sys.exit(1 if sum(total) else 0)
