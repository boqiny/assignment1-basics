# CS336 Assignment 1 · 进度追踪

最后更新：2026-08-11（Day 1 收工）

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

- [x] `num_merges` 修正为 `vocab_size - 256 - len(special_tokens)`
- [x] 合并循环（朴素版）
- [x] tie-break：`max(pair_counts, key=lambda x: (pair_counts[x], x))`
- [x] `merge()` 重建 dict
- [x] 每轮重新统计 pair_counts
- [x] `return vocab, merges`
- [ ] **修返回值标注：`tuple(...)` 写成了圆括号，应为 `tuple[...]`**（见下方 Day 2 第 0 步）
- [ ] special tokens 加入 vocab + ID 连续化
- [ ] 填 `tests/adapters.py` 的 `run_train_bpe`
- [ ] 跑通 `test_train_bpe` / `test_train_bpe_special_tokens`
- [ ] 增量更新优化（过 `test_train_bpe_speed`）

**验证：** toy 语料上 12 个 merge 与讲义完全一致 ✅

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

### Day 1 ✅ 完成 · 合并循环跑通

朴素合并循环写完，toy 上 12 个 merge 与讲义完全一致。

---

### Day 2（明天）· 测试全绿 → 优化 → TinyStories

**第 0 步（1 分钟）：修返回值标注**

```python
def train_bpe(...) -> tuple(dict[int, bytes], list[...]):
                      ^^^^^ 圆括号 → 应为方括号 tuple[...]
```

`tuple(...)` 是在**调用** tuple 构造函数，会在函数定义时就抛 `TypeError`，import 都进不去。改成方括号。

**第 1 步：special tokens 进 vocab + ID 连续化**

`test_train_bpe` 断言 `set(vocab.keys()) == set(reference_vocab.keys())`，
vocab_size=500 就必须是恰好 500 个连续 key（0~499），不能有空洞。

现在 `{0..255}` + 243 次 merge 的 `256+i` = 499 个，**缺的就是 special token 那一格**。

好消息：value 是按**集合**比的（`set(vocab.values())`），
所以 special token 放 256 还是放 499 都行，挑一个方案即可，别纠结。

**第 2 步：填 `tests/adapters.py` 的 `run_train_bpe`**

注意它要的返回值是 tuple `(vocab, merges)`，不是 `BPETokenizerParams`。

**第 3 步：跑测试**

```bash
uv run pytest tests/test_train_bpe.py
```

预期：`test_train_bpe` ✅、`test_train_bpe_special_tokens` ✅、
`test_train_bpe_speed` ❌（朴素版必挂，见下一步）

**第 4 步：增量更新优化**

`test_train_bpe_speed` 卡 1.5 秒，测试注释里明说「toy implementation 要 ~3 秒」。
瓶颈是每轮都 `count_adjacent_pairs` 重扫全部 pre-token。

核心思路：merge 之后**只有与被合并 pair 重叠的那些 pair 计数变了**。
需要一个从「pair → 包含它的 pre-token」的反向索引，
这样能直接跳到受影响的 pre-token，不用全表扫描。

⚠️ **改之前先确认朴素版全绿** —— 有正确版本当参照，优化改错了立刻能对比出来。
反过来做的话会同时调「逻辑对不对」和「优化对不对」两件事。

⚠️ 另注意：**`multiprocessing.Pool` 的启动开销也算进这 1.5 秒**。
macOS 默认 spawn，起 4 个进程要重新 import 模块。
`corpus.en` 很小，并行可能反而是负收益 —— 按文件大小决定要不要开多进程。

**第 5 步：TinyStories 10K**

跑 10K 词表，vocab / merges 序列化落盘。
回答 `train_bpe_tinystories`：耗时、内存、最长 token 是什么（合理吗）、profile 出瓶颈在哪。

**第 6 步：后台启动 OWT 32K**

做完上一步立刻挂上去（上限 12 小时），挂着去看 lecture。

**验收：** 三个 test 全绿；TinyStories 词表落盘；OWT 在跑

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

1. ~~**tie-break 在 `bytes` 上比较。**~~ ✅ 已解决，toy 上验证第一次 merge 取到了 `(b's', b't')`。

2. **special token 的包含关系。** 正则 `|` 是从左到右先匹配先算。如果 `special_tokens` 里同时有 `<|endoftext|>` 和 `<|endoftext|><|endoftext|>`，短的排前面会让长的永远匹配不全。join 之前按长度降序排。`test_tokenizer.py` 里有专门的用例。

3. **`pre_tokenize` 里的 `errors="ignore"`。** 会静默丢弃 malformed bytes，等于悄悄改了语料。目前 chunk 边界落在 special token 起始处，实际大概率不会触发。可以加个断言验证一下，或者去掉 `errors=` 看会不会真的抛。

4. **`find_chunk_boundaries` 硬编码 `b"<|endoftext|>"`。** 如果 `special_tokens` 里没有它，分块会退化成单块（不影响正确性，只影响并行度）。

5. **`count_pretokenization` 的 `num_processes` 写死为 4。** 换成 `os.cpu_count()`，另外 chunk 数可以开得比进程数多一些，方便负载均衡。

6. **内存：** OWT 是 11GB。chunk 数 = 进程数时，每个 worker 会同时持有 bytes 和解码后的 str。限制是 100GB，先跑起来看实际占用，超了再调 chunk 数。
