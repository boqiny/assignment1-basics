# CS336 Assignment 1 · 进度追踪

最后更新：2026-08-11

---

## 总原则

**BPE 不阻塞 lecture，也不阻塞 §3。**

§3 的 Transformer 模块（Linear / Embedding / RMSNorm / SwiGLU / RoPE / Attention）跟 BPE 零依赖，随时可写。BPE 唯一卡住的下游是「数据集编码成 .npy」，那是 §5 训练时才需要。

每天 BPE 封顶 2 小时，剩下时间看 lecture。

---

## 当前状态

### §2.5 Pre-tokenization ✅ 完成

- [x] `find_chunk_boundaries` 分块
- [x] `multiprocessing` 并行化，`functools.partial` 传参进 worker
- [x] special token 切分（escape 每个 token 后用 `|` join）
- [x] 空 `special_tokens` 的保护
- [x] `str` → `tuple[bytes, ...]` 转换
- [x] adjacent pair 统计（计数乘以 pre-token 频率）

**验证：** toy 语料上 pair 统计与讲义一致 ✅

### §2.4 BPE 训练 🔜 进行中

- [ ] 合并循环（朴素版）
- [ ] special tokens 加入 vocab
- [ ] `num_merges` 计算修正为 `vocab_size - 256 - len(special_tokens)`
- [ ] 返回值对齐 spec：`vocab: dict[int, bytes]`、`merges: list[tuple[bytes, bytes]]`
- [ ] 填 `tests/adapters.py` 的 `run_train_bpe`
- [ ] 增量更新优化

### §2.6 Tokenizer 类 ⬜ 未开始

- [ ] `__init__` / `from_files`
- [ ] `encode`
- [ ] `decode`（malformed bytes 用 `errors='replace'`）
- [ ] `encode_iterable`
- [ ] special tokens 处理

### 实验与简答 ⬜ 未开始

- [ ] `unicode1`（1分）
- [ ] `unicode2`（3分）
- [ ] `train_bpe_tinystories`（2分）
- [ ] `train_bpe_expts_owt`（2分）
- [ ] `tokenizer_experiments`（4分）

---

## 验证基准：`data/bpe_toy.txt`

语料是讲义 `bpe_example` 的玩具例子，改成一行一个词（避免 GPT-2 正则产生前导空格的 pre-token）。

**Step 1 — pre-token 频率（str key）**
```
{"low": 5, "lower": 2, "widest": 3, "newest": 6, "\n": 16}
```
`"\n"` 是单字节，不产生 pair，可忽略。

**Step 2 — bytes tuple**
```
{(b'l',b'o',b'w'): 5,
 (b'l',b'o',b'w',b'e',b'r'): 2,
 (b'w',b'i',b'd',b'e',b's',b't'): 3,
 (b'n',b'e',b'w',b'e',b's',b't'): 6,
 (b'\n',): 16}
```

**Step 3 — 首轮 pair 统计** ✅ 已验证
```
lo:7  ow:7  we:8  er:2  wi:3  id:3  de:3  es:9  st:9  ne:6  ew:6
```

**Step 4 — 第一次 merge**

`es` 与 `st` 均为 9，平局 → 取字典序更大的 → `(b's', b't')`

**Step 5 — merge 后**
```
{(b'l',b'o',b'w'): 5,
 (b'l',b'o',b'w',b'e',b'r'): 2,
 (b'w',b'i',b'd',b'e',b'st'): 3,
 (b'n',b'e',b'w',b'e',b'st'): 6,
 (b'\n',): 16}
```

**最终 12 个 merge**
```
s t, e st, o w, l ow, w est, n e, ne west, w i, wi d, wid est, low e, lowe r
```

跑满需要 `vocab_size = 256 + len(special_tokens) + 12`。

---

## 每日计划

### Day 1（今天）· 合并循环 → 测试全绿

1. 写朴素合并循环（每轮重新扫一遍统计 pair，不做优化）
2. special tokens 加入 vocab，`num_merges` 改对
3. 返回值改成 spec 要求的形式
4. 填 `tests/adapters.py` 的 `run_train_bpe`

**验收：**
- toy 上 12 个 merge 与上面基准完全一致
- `uv run pytest tests/test_train_bpe.py` 全绿（含 `test_train_bpe_special_tokens`）

### Day 2 · 优化 + TinyStories + 启动 OWT

1. 改增量更新：只更新受影响的 pair 计数。**改之前确认朴素版全绿**，这样出错能立刻对比
2. `cProfile` / `py-spy` 找瓶颈
3. 跑 TinyStories 10K 词表，序列化 vocab / merges 落盘
4. 回答 `train_bpe_tinystories`：耗时、内存、最长 token 是什么、瓶颈在哪
5. **做完立刻后台启动 OWT 32K 训练**（上限 12 小时），挂着去看 lecture

**验收：** 两个 test 全绿且优化前后结果一致；TinyStories 词表落盘

### Day 3 · Tokenizer 类

`encode` / `decode` / `from_files` / special tokens 处理。15 分的大头。

**验收：** `uv run pytest tests/test_tokenizer.py` 全绿

### Day 4 · 收尾

1. `encode_iterable`
2. **早上先启动数据集编码**（TinyStories + OWT → uint16 `.npy`），跑得久，边跑边写别的
3. `tokenizer_experiments`：压缩率、吞吐量、Pile 825GB 估算
4. `unicode1` / `unicode2` 简答（15 分钟能写完，卡壳时换过去写）

**验收：** §2 全部 deliverable 完成，4 个 `.npy` 就位

### Day 5 · Buffer

补漏 + 开始 §3。

---

## 已知待处理的坑

按优先级排，遇到再处理，别提前优化。

1. **tie-break 在 `bytes` 上比较，不是在 token ID 上比。** 比错了会在几十次 merge 之后才暴露，症状极难定位。写 `max` 那行时留意。

2. **special token 的包含关系。** 正则 `|` 是从左到右先匹配先算。如果 `special_tokens` 里同时有 `<|endoftext|>` 和 `<|endoftext|><|endoftext|>`，短的排前面会让长的永远匹配不全。join 之前按长度降序排。`test_tokenizer.py` 里有专门的用例。

3. **`pre_tokenize` 里的 `errors="ignore"`。** 会静默丢弃 malformed bytes，等于悄悄改了语料。目前 chunk 边界落在 special token 起始处，实际大概率不会触发。可以加个断言验证一下，或者去掉 `errors=` 看会不会真的抛。

4. **`find_chunk_boundaries` 硬编码 `b"<|endoftext|>"`。** 如果 `special_tokens` 里没有它，分块会退化成单块（不影响正确性，只影响并行度）。

5. **`count_pretokenization` 的 `num_processes` 写死为 4。** 换成 `os.cpu_count()`，另外 chunk 数可以开得比进程数多一些，方便负载均衡。

6. **内存：** OWT 是 11GB。chunk 数 = 进程数时，每个 worker 会同时持有 bytes 和解码后的 str。限制是 100GB，先跑起来看实际占用，超了再调 chunk 数。
