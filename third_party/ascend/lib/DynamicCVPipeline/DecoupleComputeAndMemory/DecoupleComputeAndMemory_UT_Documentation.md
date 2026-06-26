# DecoupleComputeAndMemory UT 说明文档



对应实现说明文档：

```text
third_party/ascend/lib/DynamicCVPipeline/DecoupleComputeAndMemory/DecoupleComputeAndMemory_Code_Documentation.md
```

## 1. 一句话概括

这组 UT 使用 MLIR lit/FileCheck 方式验证 `--gm-load-multi-buffer` pass 的核心行为：只有带 `gm_load_bufferable` 的 GM load 会被改写；合法 loop 会生成 double-buffer producer/consumer 状态机；复杂场景下多个 load、嵌套 loop、原始 iter_args、`ssbuffer.block_id` 和真实 DTS 回归都能保持正确。

## 2. 测试运行方式

每个测试文件的 RUN 行都是：

```mlir
// RUN: triton-opt --gm-load-multi-buffer %s | FileCheck %s
```

也就是说，这批 UT 直接测试 `AddMultiBufferToGMLoadPass`，而不是完整的 `DecoupleComputeAndMemoryPass` 子流水。测试输入里通常已经手工添加了：

```mlir
{gm_load_bufferable}
```

这等价于绕过 `AsyncLoadHoistingPass` 的候选识别阶段，专注验证多 buffer loop rewrite 是否正确。

## 3. 文件概览

| 文件 | 覆盖重点 |
| --- | --- |
| `test_decouple_compute_and_memory_basic_transform.mlir` | 基础 double buffer 改写、无 marker 跳过、trip count 策略、缺 backing alloc 跳过、无 marked load no-op。 |
| `test_decouple_compute_and_memory_producer_consumer.mlir` | producer IV 投影、runtime lower bound、原始 iter_args 保留、未标记 DMA 保留、flag/counter 初始化、producer 条件结构。 |
| `test_decouple_compute_and_memory_complex_scenarios.mlir` | 多 load 独立状态、嵌套 loop、线性 iter_arg delta、多个 marked load + 原 iter_args、bf16、outer/inner loop 分别改写、index IV。 |
| `test_decouple_compute_and_memory_gm_load_multi_buffer_block_id.mlir` | 大体量真实/近真实 DTS 回归，重点验证 `ssbuffer.block_id` 传播、连续 block run、多 scope、多 load、多 loop 场景。 |

## 4. 测试关注的核心 IR 特征

这些 UT 不是比较完整输出 IR，而是用 FileCheck 锁定关键结构：

1. **buffer slot 分配**
   - `memref.alloc()` 出现在 loop 前。
   - depth=2 时，每个 marked load 生成两个 slot。
   - 多个 marked load 时，每个 load group 独立生成 slot。

2. **扩展 scf.for iter_args**
   - 每个 load group 追加 `i1, i1, index, index`。
   - 对应含义是两个 slot flag、producer counter、consumer counter。
   - 原始 iter_args 必须保留在前，多 buffer 控制参数追加在后。

3. **producer 填充路径**
   - 生成 `scf.if -> (i1, index)`。
   - 条件包含 `flag == false` 和 `prodCounter < tripCount`。
   - producer 内重放地址生成和 `memref.copy`，目标是对应 slot buffer。

4. **consumer 选择路径**
   - 生成 `arith.remui` 计算 `consCounter % depth`。
   - 每个 slot 生成 `bufferization.to_tensor`。
   - 通过 `arith.select` 选择当前消费 slot。

5. **slot release**
   - 消费后生成 `scf.if -> (i1)` 清 flag。
   - `consCounter` 在 yield 中递增。

6. **保守跳过**
   - 无 `gm_load_bufferable` 不改写。
   - 静态 trip count `<= depth` 不改写。
   - 找不到 backing `memref.alloc` 不改写。
   - 没有 marked load 时 pass 为 no-op。

7. **block_id 保持**
   - 新生成的 alloc、producer if、copy、remui、to_tensor、select、flag clear 等需要带合理 `ssbuffer.block_id`。
   - 真实回归中要求顶层 block_id run 不被拆成反复交错的顺序。

## 5. Basic Transform 测试文件

文件：

```text
test_decouple_compute_and_memory_basic_transform.mlir
```

### TC-B01: Basic double buffer transformation

验证最小正向改写：

1. 一个带 `gm_load_bufferable` 的 Q load。
2. pass 在 loop 前生成两个 `memref.alloc` slot。
3. 生成 `arith.ceildivui` 计算 trip count。
4. `scf.for` 结果类型扩展为：

```text
tensor<128x128xf32>, i1, i1, index, index
```

5. producer 分别向 slot0/slot1 `memref.copy`。
6. consumer 用 `arith.remui`、`to_tensor`、`arith.select` 选择 slot。
7. 原 `linalg.matmul` 使用 selected tensor。

对应实现点：

- `allocateBufferSlots()`
- `emitLoopInvariantValues()`
- `buildExtendedFor()`
- `emitProducerFillForSlot()`
- `emitLoadSlotSelection()`

### TC-B02: No marker pass-through

验证没有 `gm_load_bufferable` 时不改写：

1. 不出现 `arith.ceildivui`。
2. 不出现 `arith.remui`。
3. 不出现 `arith.select`。
4. 原 `scf.for` 和 `memref.copy` 保留。

对应实现点：

- `collectMarkedOps()` 只收集带 marker 的 op。
- `collectAndGroupMarkedOps()` 在 `markedOps_` 为空时直接返回。

### TC-B03: Trip count <= depth skip

验证静态 trip count 等于 depth 时跳过。

输入 loop：

```text
0 to 2 step 1
```

trip count 为 2，默认 double buffer depth 为 2，因此无流水收益，pass 不应改写。

对应实现点：

- `getConstantTripCount()`
- `collectAndGroupMarkedOps()` 中 `tripCount <= depth` 的过滤逻辑。

### TC-B03b: Trip count = depth + 1 transform

验证边界条件：trip count 为 3，大于 depth=2，应该触发改写。

FileCheck 观察：

1. 出现 `arith.ceildivui`。
2. 出现 `arith.remui ..., %c2`。

意义是防止过滤条件写成 `tripCount < depth` 或错误跳过最小有效循环。

### TC-B13: No backing alloc skip

验证 marked `to_tensor` 的依赖链中没有 `memref.alloc` 时跳过。

输入直接对 `reinterpret_cast` 做 `bufferization.to_tensor`，没有中间 alloc。pass 无法创建“原 alloc -> slot buffer”的映射，因此必须保守跳过。

对应实现点：

- `computeLoadChain()`
- `findBackingAlloc()`
- `collectMarkedOps()` 找不到 alloc 时不加入结果。

### TC-B15: No marked loads no-op

验证完全没有 marked load 的空/普通 loop 不被改写。

这是 pass 稳定性测试，防止空输入下误生成多 buffer 控制结构。

## 6. Producer/Consumer 测试文件

文件：

```text
test_decouple_compute_and_memory_producer_consumer.mlir
```

### TC-B04: Producer IV projection correctness

验证 producer 不是使用当前 loop IV，而是计算虚拟 IV：

```text
projectedIV = lowerBound + prodCounter * step
```

FileCheck 关注：

1. `index_cast index -> i32`
2. `muli prodCounter, step`
3. `addi lowerBound, offset`
4. 再 cast 回 index 参与地址计算
5. 最终执行 `memref.copy`

对应实现点：

- `mapProducerLoopPosition()`

### TC-B04b: IV projection with runtime lower bound

验证 lower bound 是运行时参数 `%offset_init` 时，producer 仍然用该值参与投影，而不是假设 lower bound 为常量 0。

这个测试覆盖动态 lower bound，防止地址预取在非 0 起点 loop 中错误。

### TC-B06: Preserve original iter_args

验证原 loop 已有两个 iter_args：

```text
%acc, %lse
```

改写后类型顺序应为：

```text
tensor<128x128xf32>, tensor<128xf32>, i1, i1, index, index
```

意义：

1. 原有 loop-carried values 必须保留。
2. 多 buffer 控制状态只能追加，不能改变原 result 顺序。

对应实现点：

- `buildExtendedFor()`
- `replaceOriginalForResults()`
- `emitCombinedYield()`

### TC-B07: Dead DMA elimination

输入中有 Q load 和 K load：

1. Q 带 `gm_load_bufferable`，应改写为多 buffer。
2. K 没有 marker，应保留在 consumer body 中正常 alloc+copy。
3. Q 原 consumer body 中的 DMA/copy 链应被跳过，改为使用 selected tensor。

该测试验证“只消除 marked load 的原始 DMA，不影响未标记 load”。

对应实现点：

- `computeSkipInConsumer()`
- `shouldSkipOldBodyOp()`
- `cloneOldBodyOpIfNeeded()`

### TC-B16: Producer flag state machine initialization

验证新增控制 iter_args 的初值：

```text
flag0 = false
flag1 = false
prodCounter = 0
consCounter = 0
```

对应实现点：

- `ConstantCache::getFalse()`
- `ConstantCache::getIndex(..., 0)`
- `buildExtendedFor()`

### TC-B17: Producer condition structure

验证 producer 填 slot 的条件是：

```text
flag == false && prodCounter < tripCount
```

FileCheck 关注：

1. `arith.cmpi eq`
2. `arith.cmpi ult`
3. `arith.andi`

对应实现点：

- `createProducerFillCondition()`

## 7. Complex Scenarios 测试文件

文件：

```text
test_decouple_compute_and_memory_complex_scenarios.mlir
```

### TC-B10: Multiple loads in same loop

验证同一个 loop 中两个 marked load 独立管理状态：

1. 两个 load，各自两个 slot，总共 4 个 alloc。
2. 每个 load group 追加 4 个控制 iter_args，总共 8 个控制参数。
3. 两个 load 各有自己的 slot selection `arith.select`。

对应实现点：

- `groupByEnclosingForOp()` 中“一 marked load 一 LoadGroup”的设计。
- `buildExtendedFor()` 对每个 group 追加独立 flags/counters。

### TC-B11: Nested loops inner first

验证只有 inner loop 有 marked load 时：

1. outer loop 保持普通 loop 结构。
2. inner loop 被改写。
3. inner loop 的 buffer slots 分配在 outer body 内、inner loop 前。

对应实现点：

- `sortContextsInnerFirst()`
- `applyMultiBufferToForLoop()`
- cleanup 阶段对嵌套 forOp 的处理。

### TC-B12: Linear iter_arg delta projection

验证 loop-carried iter_arg 的线性更新可被 producer 识别：

```mlir
%offset_new = arith.addi %offset, %c128_i32
scf.yield %offset_new
```

pass 应识别 delta 为 `%c128_i32`，producer 侧可按：

```text
projectedOffset = initOffset + prodCounter * 128
```

进行地址投影。

对应实现点：

- `getLinearIterArgDelta()`
- `collectLinearIterArgDeltas()`
- `mapProducerIterArgs()`

### TC-B18: Multiple marked loads + original iter_args

组合 TC-B06 和 TC-B10：

1. 原始 iter_args 有 `%acc`、`%lse`。
2. 同一 loop 有 Q/K 两个 marked load。
3. 改写后 result 类型顺序应为：

```text
orig args 2 个 + load group 1 控制 4 个 + load group 2 控制 4 个
```

该测试主要防止多 load 与原 iter_args 组合时 result/yield 顺序错乱。

### TC-B19: Different data types bf16

验证 pass 不依赖 f16，bf16 memref/tensor 也能正确生成 slot：

```text
memref<128x128xbf16>
```

FileCheck 关注 bf16 slot alloc 和 `arith.remui`。

### TC-B20: Nested loops with marked load in outer loop only

验证只有 outer loop 有 marked load 时：

1. outer loop 被改写。
2. buffer slots 出现在 outer loop 前。
3. inner loop 没有 marked load，应保持原结构。

这和 TC-B11 互补，覆盖“inner only”和“outer only”两种嵌套情况。

### TC-B21: Index-typed IV

验证 `scf.for` induction var 类型是 index 时也能改写。

意义：

1. producer 投影不应假设 IV 一定是 i32。
2. 地址链使用 index arithmetic 时不应产生错误 cast。

对应实现点：

- `castIndexTo()`
- `castToIndex()`
- `mapProducerLoopPosition()`

## 8. Block ID 回归测试文件

文件：

```text
test_decouple_compute_and_memory_gm_load_multi_buffer_block_id.mlir
```

这是最大的一组回归测试，输入更接近真实 DTS 生成 IR，含 `hivm`、`scope.scope`、`ssbuffer.block_id`、`transfer_id`、VECTOR/CUBE scope、nested loop 等复杂结构。

### Test Group B1: DTS1 VECTOR shared address

函数：

```mlir
func.func @_attn_bwd
```

覆盖点：

1. block 7 和 block 8 各有一个 marked load。
2. 两个 load 共享部分地址计算，但必须有独立 buffer slots、flags、producer counter 和 consumer counter。
3. 2 个原始 iter_args + 2 个 load group 的 8 个控制 iter_args，总共 10 个 result/yield 值。
4. 新生成 producer/consumer/release 结构必须分别带 block 7、block 8 的 `ssbuffer.block_id`。
5. 顶层 block_id run 必须保持合理顺序，避免旧问题中的：

```text
7 -> 8 -> 7 -> 8 -> 9
```

目标是形成更连续的：

```text
7 -> 8 -> 9
```

这个 case 主要对应实现里的：

- `computeSkipInConsumer()`
- `emitOldBodyWithMultiBufferHooks()`
- `tagWithBlockId()`
- `logRepeatedTopLevelBlockRuns()`

### Test Group B2: Complex multi-buffer with inner alloc/fill/copy

函数：

```mlir
func.func @_attn_bwd_complex
```

覆盖点：

1. 同一个 loop 中两个 marked load，类型不同：
   - `memref<64x32xbf16>`
   - `memref<64xf32>`
2. 每个 load double buffer，总共 4 个 slot。
3. 原 loop 有 2 个 tensor iter_args，改写后追加 8 个控制 iter_args。
4. producer 侧不仅要 clone 简单地址计算，还要处理 inner alloc、fill、subview、copy 等复杂依赖链。
5. block 13 的 producer、consumer select、flag clear 都要保持 `ssbuffer.block_id = 13`。
6. block 14 的后续 compute 仍应保留并正确接上 selected tensor。

这个 case 主要回归：

- `computeLoadChain()` 四阶段依赖链收集。
- `forwardTraceFromAllocs()` 对 alloc 用户的追踪。
- `captureRegionFreeVars()` 对 region 内自由变量的追踪。
- `materializeProducerChain()` 对复杂链的 producer 侧重放。

### Test Group B3: attn_fwd mixed VECTOR + CUBE scopes

函数：

```mlir
func.func @_attn_fwd
```

覆盖点：

1. Scope 1 是 VECTOR，没有 `gm_load_bufferable`，应保持不变。
2. Scope 2 是 CUBE，含 3 个 marked load：
   - Load A：outer loop，`block_id=2`
   - Load B：inner loop，`block_id=0`
   - Load C：inner loop，`block_id=1`
3. outer loop 为 Load A 追加 4 个控制 iter_args。
4. inner loop 为 Load B/C 追加 8 个控制 iter_args。
5. outer 和 inner 的 yield 类型分别检查：

```text
outer: i1, i1, index, index
inner: i1, i1, index, index, i1, i1, index, index
```

6. block 0/1/2 的 producer copy、consumer select、flag clear 都需要保持各自 block id。

这个 case 主要覆盖：

- 嵌套 loop inner-first 改写。
- 同一函数内多个 scope 的选择性改写。
- 多个 block_id 的独立状态机生成。

## 9. UT 与实现代码对应关系

| 实现模块 | 主要 UT 覆盖 |
| --- | --- |
| `collectMarkedOps()` | TC-B02、TC-B15、block_id B3 Scope 1 no-op。 |
| `findBackingAlloc()` | TC-B13。 |
| `getConstantTripCount()` 和 depth 过滤 | TC-B03、TC-B03b。 |
| `groupByEnclosingForOp()` | TC-B10、TC-B18、block_id B1/B2/B3。 |
| `sortContextsInnerFirst()` | TC-B11、TC-B20、block_id B3。 |
| `allocateBufferSlots()` | TC-B01、TC-B10、TC-B19、block_id B1/B2/B3。 |
| `buildExtendedFor()` | TC-B01、TC-B06、TC-B10、TC-B18、block_id B1/B2/B3。 |
| `mapProducerLoopPosition()` | TC-B04、TC-B04b、TC-B21。 |
| `getLinearIterArgDelta()` / `mapProducerIterArgs()` | TC-B12。 |
| `createProducerFillCondition()` | TC-B17。 |
| `emitLoadSlotSelection()` | TC-B01、TC-B10、block_id B1/B2/B3。 |
| `computeSkipInConsumer()` | TC-B07、block_id B1。 |
| `tagWithBlockId()` | block_id B1/B2/B3。 |
| `emitCombinedYield()` | TC-B06、TC-B18、block_id B1/B2/B3。 |

## 10. 覆盖矩阵

| 能力点 | 覆盖用例 |
| --- | --- |
| 基础 double buffer | TC-B01 |
| 无 marker no-op | TC-B02、TC-B15、B3 Scope 1 |
| trip count 过滤 | TC-B03、TC-B03b |
| 缺 backing alloc 跳过 | TC-B13 |
| producer IV 投影 | TC-B04、TC-B04b |
| 原始 iter_args 保留 | TC-B06、TC-B18 |
| marked DMA 从 consumer 删除 | TC-B07 |
| flag/counter 初始化 | TC-B16 |
| producer 条件 | TC-B17 |
| 多 marked load 独立状态 | TC-B10、TC-B18、B1、B2、B3 |
| 嵌套 loop | TC-B11、TC-B20、B3 |
| 线性 iter_arg delta | TC-B12 |
| bf16 类型 | TC-B19、B2 |
| index IV | TC-B21 |
| block_id 传播与连续 run | B1、B2、B3 |
| 真实 DTS 复杂 IR 回归 | B1、B2、B3 |



## 13. 后续补充测试建议

当前 UT 已覆盖多 buffer 改写主路径和多个回归点。后续如果扩展实现，建议补充：

1. `AsyncLoadHoistingPass` 的独立 UT：覆盖自动添加 `gm_load_bufferable` 的识别逻辑。
2. 非线性 `reinterpret_cast` offset 被拒绝的测试。
3. region free var 更小型的最小复现测试，避免只依赖大 DTS 文件覆盖。
4. 动态 trip count 下仍可改写的测试。
5. depth 不等于 2 时的参数化或专门测试，例如 depth=3 的 slot/select/yield 结构。
