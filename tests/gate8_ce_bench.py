import os, time, torch, torch.multiprocessing as mp, torch.distributed as dist
os.environ.setdefault('MASTER_ADDR','127.0.0.1'); os.environ.setdefault('MASTER_PORT','29543')
S, H, I = 1024, 5120, 17408
SEGB = S*H*2
def run(rank):
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo', rank=rank, world_size=2)
    import sglang.srt.distributed.device_communicators.mailbox_all_reduce as m
    ar = m.MailboxAR(None, rank)
    ops = ar.ops
    torch.cuda.synchronize()
    x = torch.randn(S, I, dtype=torch.bfloat16, device='cuda')
    w = torch.randn(I, H, dtype=torch.bfloat16, device='cuda')
    src = torch.randn(S*H, dtype=torch.bfloat16, device='cuda')
    dst = ar.peer_mbx + 256 + (1<<20)
    s2 = torch.cuda.Stream()
    e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True); ev = torch.cuda.Event(True)
    def bench(fn, iters=30, warm=5):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); dist.barrier()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
            ts.append((time.perf_counter()-t0)*1e3)
        return sorted(ts)[iters//2]
    A = bench(lambda: torch.matmul(x, w))
    # B: CE push 单独
    B = bench(lambda: ops.mailbox_ce_push(dst, src.data_ptr(), SEGB, 0))
    print(f'r{rank} A gemm={A:.3f} B ce_push={B:.3f} bw={SEGB/1e6/ (B/1000) / 1.024:.2f} GB/s', flush=True)
    # C: 串行 GEMM -> CE push
    C = bench(lambda: (ops.mailbox_ce_push(dst, src.data_ptr(), SEGB, 0), torch.matmul(x, w)))
    # D: 并行 — 副流 CE push + 主流 GEMM
    for _ in range(5):
        ev.record(); s2.wait_event(ev); ops.mailbox_ce_push(dst, src.data_ptr(), SEGB, s2.cuda_stream)
        torch.matmul(x, w); torch.cuda.synchronize()
    dist.barrier()
    ts_d, ts_g, ts_p = [], [], []
    p0 = torch.cuda.Event(True); p1 = torch.cuda.Event(True)
    for _ in range(30):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ev.record(); s2.wait_event(ev)
        p0.record(s2); ops.mailbox_ce_push(dst, src.data_ptr(), SEGB, s2.cuda_stream); p1.record(s2)
        e0.record(); torch.matmul(x, w); e1.record()
        torch.cuda.synchronize()
        ts_d.append((time.perf_counter()-t0)*1e3); ts_g.append(e0.elapsed_time(e1)); ts_p.append(p1.elapsed_time(p0) - p0.elapsed_time(p1)*0) 
    ts_d.sort(); ts_g.sort()
    # p 时间窗记录简化: 只看 D 总与 gemm 段
    print(f'r{rank} C serial={C:.3f} D par={ts_d[15]:.3f} D.gemm={ts_g[15]:.3f} gemm_drop={100*(ts_g[15]/A-1):.1f}% overlap_gain={(C-ts_d[15]):.3f}ms', flush=True)
    dist.destroy_process_group()
if __name__ == '__main__':
    mp.spawn(run, args=(), nprocs=2)
