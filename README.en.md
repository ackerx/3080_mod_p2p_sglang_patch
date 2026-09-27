# mailbox-allreduce — True P2P AllReduce for consumer-GPU TP=2 without NVLink

[English](README.en.md) | [中文](README.md)

Run allreduce over a real PCIe BAR1 direct path (10.26 GB/s, ≈1.36× NCCL on the
same machine) for two NVIDIA GeForce cards that officially have P2P disabled —
inside sglang TP=2 inference, with native decode-CUDA-graph compatibility.

Three layers:

1. **Kernel-module patch** (open-gpu-kernel-modules 595.91.07): enables BAR1 P2P
   on GeForce and adds a dual-window BAR1 mapping mode — the hardware
   prerequisite for a 64MB mailbox to land in the window reliably;
2. **User-space mailbox protocol** (JIT-compiled torch extension): a 5-kernel
   lockstep push allreduce that DMAs its input straight into the peer GPU's
   "mailbox" in VRAM, where the peer reduces locally;
3. **sglang communicator**: implements the same interface surface as
   CustomAllreduce — 3 lines of wiring, enabled with `SGLANG_MAILBOX_AR=1`,
   falls back to NCCL automatically when out of range.

## Why not NCCL / CustomAllreduce

On consumer GPUs `cudaDeviceCanAccessPeer = 0`: NCCL falls back to SHM via the
host (two PCIe crossings per byte, ~6.9 GB/s; forcing the P2P transport hangs
the handshake), and CustomAllreduce needs P2P buffer registration, which fails
the same way. This project enables P2P at the driver level, then sidesteps the
whole "allocate and register a buffer per AR" path with a fixed 64MB mailbox —
the mailbox address never changes, so CUDA-graph kernels record raw pointers
and decode replay costs nothing extra.

## Performance summary (measured on 2× RTX 3080-20G / PCIe 3.0 x16)

| Path | Effective AR bandwidth |
|---|---|
| NCCL (consumer default: SHM via host) | ~6.9 GB/s |
| NCCL (same driver, same machine) | 8.86 GB/s |
| **This project (10MB~24MB)** | **9.15~9.28 GB/s** (raw push link 10.26; full duplex 19.3~20.1) |

End-to-end gain = AR share × link advantage: allreduce is ~21% of prefill
(chunk 1024) and only 2~3% of decode ITL — **this is a "last mile" optimization
of the communication stack, not an order-of-magnitude win**; the biggest
benefit comes from migrating away from NCCL's SHM default. For the full
picture — layer-by-layer bandwidth decomposition, measurement discipline
(cross-day comparisons will lie to you), and why compute×communication overlap
doesn't pay off on real AWQ models — see
[docs/performance.md](docs/performance.md).

## Requirements

- Two cards on the same root complex (direct PCIe attach, no NUMA hop);
- Linux, the open kernel module stack (verified on 595.91.07), root access;
- sglang 0.5.19 (other versions need the communicator interface re-aligned);
- Acceptance of an "experiment box" profile: a patched driver resident, and a
  rare kernel deadlock recovering only by rebooting the machine.

## Quick start

```bash
# 1. Driver: build & switch to the patched module (FLR + readiness probing in the script)
git clone --branch 595.91.07 --depth 1 https://github.com/NVIDIA/open-gpu-kernel-modules.git
cd open-gpu-kernel-modules && git apply /path/to/driver/p2p-mailbox-595.91.07.patch
cd kernel-open && make modules -j && sudo insmod nvidia.ko   # see driver/README.md

# 2. Bare-link validation (two terminals, CUDA_VISIBLE_DEVICES=0/1)
python tests/test_mailbox_torch.py

# 3. Wire into sglang (two files + 3 lines) and launch
SGLANG_MAILBOX_AR=1 python -m sglang.launch_server --model <model> --tp 2
# Expect in the log: [MailboxAR] ready rank=0 attempt=0 ...
```

Full deployment (FLR reset, mailbox-window troubleshooting, watchdog):
[docs/deploy.md](docs/deploy.md) (Chinese; the READMEs under `driver/` and
`sglang/` are annotated in Chinese but the commands are copy-pasteable).

## Layout

```
├── driver/     driver patch (GA102 + dual-window BAR1) + build guide
├── sglang/     mailbox_ar_ext.cu + mailbox_all_reduce.py + wiring guide
├── tests/      bare-link smoke / 560-round hammer / size sweep / overlap gate bench / pp A/B
├── scripts/    one-shot start-stop (driver swap + FLR + watchdog) / D-state watchdog
└── docs/       principle / deploy / performance / limitations (Chinese)
```

## Docs

- [Principle & design](docs/principle.md) — three-layer structure, mailbox
  layout, lockstep protocol, and the microbenchmark evidence behind every
  engineering decision
- [Deploy guide](docs/deploy.md) — three independently verifiable layers,
  troubleshooting quick reference
- [Performance](docs/performance.md) — bandwidth decomposition, end-to-end
  accounting, measurement discipline
- [Limitations](docs/limitations.md) — do the math first: where the gains end

## License & acknowledgments

- User-space code (sglang/, tests/, scripts/, docs/): MIT License
- The driver patch follows upstream open-gpu-kernel-modules licensing
  (kernel-open: MIT; src/nvidia: NVIDIA Open Source License)
- The P2P-enabling approach builds on community work: the
  [tinygrad 550.54.15-p2p](https://github.com/tinygrad/open-gpu-kernel-modules/tree/550.54.15-p2p)
  branch and Aikitoria's 595-p2p patch; this repo's delta (GA102 support +
  dual-window BAR1) extends on top of them

## Status

Experimental open source: verified on a single machine, TP=2 only, no CI.
Reproductions on other consumer-GPU combinations are very welcome — please
report your numbers.
