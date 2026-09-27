# pp_bench.py — 真机 prefill 吞吐(pp tok/s)基准: max_tokens=1 隔离 prefill,
# 不同 salt 文本防 prefix cache 命中。pp = usage.prompt_tokens / 端到端 wall。
# 用法: python pp_bench.py <模式>   模式: full=128K×2+40K×2, warm=仅预热
import json
import sys
import time

import requests

BASE = "http://127.0.0.1:8001"


def bench(tag, chars, salt):
    sent = f"第{salt}组：大模型推理优化涵盖内核融合、通信重叠、显存管理与调度策略。"
    text = sent * (chars // len(sent) + 1)
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": text + "\n\n只回复:收到"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    t0 = time.time()
    r = requests.post(BASE + "/v1/chat/completions", json=payload, timeout=600)
    dt = time.time() - t0
    j = r.json()
    if r.status_code != 200:
        print(f"{tag}: HTTP {r.status_code} {str(j)[:200]}")
        return None
    pt = j["usage"]["prompt_tokens"]
    pp = pt / dt
    print(f"{tag}: prompt={pt} wall={dt:.1f}s pp={pp:.0f} tok/s", flush=True)
    return pp


mode = sys.argv[1] if len(sys.argv) > 1 else "full"
bench("warmup", 2000, 0)
if mode == "warm":
    sys.exit(0)
res = {}
for tag, chars, salt in (("128K-1", 206000, 1), ("128K-2", 204500, 2),
                         ("40K-1", 62000, 3), ("40K-2", 63500, 4)):
    res[tag] = bench(tag, chars, salt)
vals = [v for v in res.values() if v]
print("SUMMARY " + json.dumps({k: (round(v) if v else None) for k, v in res.items()}),
      flush=True)
