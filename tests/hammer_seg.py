# hammer_seg.py — #8 分段 AR 压力锤子: 连续段 AR(模拟逐层 down_proj) + fb 混跑
# 用法: 双 rank 同时起 (CUDA_VISIBLE_DEVICES=<rank>, argv[1]=rank, argv[2]=轮数)
# seg 归约算式与 fb 完全一致(out=local+peer 单次加) → 结果应 bit 级相等
import os
import sys
import time

import torch
import torch.distributed as dist

os.environ.setdefault("SGLANG_MAILBOX_AR", "1")
rank = int(sys.argv[1])
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29589",
                        rank=rank, world_size=2)
dev = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else rank
torch.cuda.set_device(dev)
from sglang.srt.distributed.device_communicators.mailbox_all_reduce import MailboxAR

ar = MailboxAR(None, dev)
assert not ar.disabled, "MailboxAR 未落窗"
print(f"[r{rank}] ready", flush=True)

torch.manual_seed(99 + rank)
H = 5120
S = 1024                      # down_proj 场景: [1024, 17408]@[17408,5120] → AR[1024,5120]
seg_stream = torch.cuda.Stream()
fails = 0
t_seg = 0.0
t_fb = 0.0

for rep in range(ROUNDS):
    x = torch.randn(S, H, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x)
    s0 = time.perf_counter()
    seq1 = ar.submit_seg(x[:512], seg_stream)
    seq2 = ar.submit_seg(x[512:], seg_stream)
    ar.wait_seg(seq1, x[:512], out[:512])
    ar.wait_seg(seq2, x[512:], out[512:])
    torch.cuda.synchronize()
    t_seg += time.perf_counter() - s0
    e = ar.seg_err()
    if e != 0:
        fails += 1
        print(f"[r{rank}] rep{rep} seg_err={e}", flush=True)
        break
    s0 = time.perf_counter()
    fb = ar.custom_all_reduce(x)
    torch.cuda.synchronize()
    t_fb += time.perf_counter() - s0
    if fb is None:
        fails += 1
        print(f"[r{rank}] rep{rep} fb disabled", flush=True)
        break
    if not torch.equal(out, fb):
        fails += 1
        mx = (out.float() - fb.float()).abs().max().item()
        print(f"[r{rank}] rep{rep} MISMATCH max_err={mx}", flush=True)
        break
    if (rep + 1) % 50 == 0:
        print(f"[r{rank}] {rep + 1}/{ROUNDS} pass", flush=True)

dist.barrier()
tot = [0, 0]
dist.all_gather_object(tot, fails)
print(f"[r{rank}] seg_avg={t_seg / max(ROUNDS, 1) * 1000:.3f}ms "
      f"fb_avg={t_fb / max(ROUNDS, 1) * 1000:.3f}ms "
      f"RESULT fails={sum(tot)}", flush=True)
sys.exit(1 if sum(tot) else 0)
