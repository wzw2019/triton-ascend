# DecoupleComputeAndMemory 代码分析文档

本文档面向代码述职和后续维护，分析范围包括：

- 入口文件：`third_party/ascend/lib/DynamicCVPipeline/DecoupleComputeAndMemory.cpp`
- 核心实现目录：`third_party/ascend/lib/DynamicCVPipeline/DecoupleComputeAndMemory/`
- 相关头文件：`third_party/ascend/include/DynamicCVPipeline/DecoupleComputeAndMemory/`

## 1. 一句话概括

`DecoupleComputeAndMemory` 是 Dynamic CV Pipeline 中用于“计算与访存解耦”的 MLIR pass 子流水。它先识别适合提前调度的 GM load，再把这些 load 所在的 `scf.for` 改写成多 buffer 生产者/消费者结构，从而让下一批数据的搬运和当前批数据的计算在循环内形成重叠。

## 2. 解决的问题与收益

原始 IR 中，访存和计算通常串行出现在循环体内：

```mlir
scf.for %i = %lb to %ub step %step {
  %buf = memref.alloc ...
  ... address generation ...
  %tensor = bufferization.to_tensor %buf
  ... compute using %tensor ...
}
```

这种结构的问题是：

1. 每轮循环都要等本轮 GM load 完成后才能进入后续计算。
2. 访存延迟没有被隐藏，Cube/Vector 计算单元可能等待数据。
3. 单 buffer 无法表达“当前消费一个槽，同时提前填充另一个槽”的流水行为。

该 pass 的目标是把循环改写为：

```text
for each iteration:
  producer: 找空 slot，按 producerCounter 对应的未来循环位置生成地址并填充 buffer
  consumer: 按 consCounter % depth 选择当前应消费的 slot
  compute: 使用选中的 tensor 执行原计算
  release: 消费完成后清空 slot flag，consCounter++
```

收益是将 GM load 与计算解耦，用 `depth` 个 buffer slot 形成环形队列，提升 load/compute overlap 的机会。

## 3. 总体流水

入口位于 `DecoupleComputeAndMemoryPass::runOnOperation()`：

```cpp
int depth = BufferCountManager::getInstance()
  .getBufferCountByType(BufferCountManager::DepType::LoadStore);

if (depth <= 1)
  return;

pm.addPass(createAsyncLoadHoistingPass());
pm.addPass(createAddMultiBufferToGMLoadPass());
runPipeline(pm, module);
```

关键点：

1. 由 `BufferCountManager` 决定 LoadStore 依赖类型的 buffer 深度。
2. `depth <= 1` 时不做多 buffer 改写，因为单 buffer 无法形成有效流水。
3. 子流水包含两个 pass：
   - `AsyncLoadHoistingPass`：识别并标记可 buffer 化的 load。
   - `AddMultiBufferToGMLoadPass`：对标记 load 所在循环做多 buffer 改写。

## 4. 目录与文件职责

| 文件 | 职责 |
| --- | --- |
| `DecoupleComputeAndMemory.cpp` | 顶层 pass 入口，读取 buffer depth 并组织子 pass pipeline。 |
| `AsyncLoadHoisting.cpp` | 扫描 IR，识别可进行 GM load 多 buffer 的候选 load，并添加 `gm_load_bufferable` 标记。 |
| `AddMultiBufferToGMLoad.cpp` | 多 buffer pass 的顶层调度：收集标记、按 loop 分组、排序、执行改写、清理 IR。 |
| `DependencyAnalysis.cpp` | 计算 marked load 的依赖链，按最近 `scf.for` 分组，并分析 consumer 侧哪些链上 op 可跳过。 |
| `LoopTransform.cpp` | 创建新 buffer slot、扩展 `scf.for` 的 iter_args、计算 trip count、清理死 iter_arg 和死 op。 |
| `MultiBufferLoopBodyEmission.cpp` | 生成多 buffer 循环体，包括 producer 填充、consumer slot 选择、原 body 克隆、slot release 和 yield。 |
| `AddMultiBufferToGMLoadTypes.h` | 定义 `MarkedLoad`、`LoadGroup`、`ForBufferCtx`、`ExtendedForInfo` 等核心数据结构。 |
| `AddMultiBufferToGMLoadInternal.h` | 内部常量、debug 宏、工具函数声明和 `ConstantCache`。 |

## 5. 关键属性与状态

### 5.1 IR 属性

| 属性 | 含义 |
| --- | --- |
| `gm_load_bufferable` | `AsyncLoadHoistingPass` 标记出的可多 buffer 化 load。后续 pass 只处理带该标记的 op。 |
| `ssbuffer.block_id` | 上游或本 pass 用于标识一段 SSBuffer 相关 IR 的 block id。标记、过滤和 debug 都依赖它。 |
| `ssbuffer.load_store` | 给新生成的 producer/release `scf.if` 添加的访存相关属性。 |

### 5.2 多 buffer 状态

每个 `LoadGroup` 会向新 `scf.for` 增加：

```text
flags[depth]       // 每个 slot 是否已经被 producer 填充
prodCounter        // producer 已经预取到第几轮逻辑迭代
consCounter        // consumer 当前消费到第几轮逻辑迭代
```

这些状态全部以 `iter_args` 的形式穿过 `scf.for`，保证 SSA 形式下跨迭代传递。

## 6. AsyncLoadHoistingPass 分析

该 pass 的作用不是直接移动 op，而是筛选可被后续多 buffer pass 处理的 load，并给它们打上 `gm_load_bufferable`。

### 6.1 load 识别规则

`isLoadOp(Operation *op)` 基于 `MemoryEffectOpInterface` 判断是否有 `MemoryEffects::Read`，同时排除一些不适合改写的情况：

1. 排除 `memref.copy`。
2. 排除所有 index 都是常量的标量 `memref.load`，这类通常不是主 GM load 数据通路。
3. 排除只被 mask processing 使用的 op，例如 `arith.cmpf`、`arith.select`、`arith.maximumf`、`arith.minimumf` 等链路。
4. 必须有 read side effect。

### 6.2 地址生成链分析

`getAddressGenerationChainWithFilter()` 从 load 反向收集地址生成依赖：

1. 从 load 的 operand 出发递归追踪 defining op。
2. 跳过常量和另一个 load。
3. 对 memref 类型 operand，额外追踪 `memref.copy`、`memref.subview` 等用户关系。
4. 收集后按照 block 中原始顺序稳定排序。
5. 使用 `ssbuffer.block_id` 过滤出与 load 同 block id 的链。

### 6.3 合法性过滤

候选 load 需要满足：

1. 依赖链中存在 block argument：`chainContainsBlockArg()`。这说明地址或数据访问与循环位置相关，适合做跨迭代投影。
2. `memref.reinterpret_cast` 的 offset 必须线性可预测：`filterNonLinearReinterpretCast()`。
3. 线性检查会拒绝 `memref.load` 间接寻址、`scf.while`、嵌套 `scf.for` 等不可简单投影的控制/数据流。

### 6.4 按 block_id 连续段扫描

`asyncLoadHoistingImpl()` 会递归遍历函数体 region，并按连续的 `ssbuffer.block_id` run 缓存 op。每遇到 block id 改变，就对上一段调用 `scanAndHoistBlock()`。

这样做的意义是：

1. 保证同一 block id 的相关 op 在同一段内一起分析。
2. 避免跨 block id 错误合并地址链。
3. 为后续 `AddMultiBufferToGMLoadPass` 提供精确 marker。

## 7. AddMultiBufferToGMLoadPass 顶层流程

`AddMultiBufferToGMLoadPass::runOnOperation()` 分为四步。

### 7.1 收集与分组

`collectAndGroupMarkedOps()`：

1. `collectMarkedOps(module)` 扫描全 module，找到带 `gm_load_bufferable` 的 op。
2. 为每个 marked op 计算完整依赖链 `computeLoadChain()`。
3. 找到 backing `memref.alloc`，用于后续替换成多 buffer slot。
4. `groupByEnclosingForOp()` 按最近的 `scf.for` 分组。
5. 每个 marked load 独立成为一个 `LoadGroup`，每组有自己的 flag 和 counter。
6. 如果静态 trip count `<= depth`，跳过该 loop，因为循环轮数不足以摊销多 buffer 成本。

### 7.2 内层 loop 优先

`sortContextsInnerFirst()` 按 loop 嵌套深度降序排序。原因是：

1. 先改写内层 loop，外层 clone body 时可以识别这些已处理的 loop。
2. 避免外层改写时把原始内层 loop 又复制一份，造成重复改写或 SSA 混乱。

### 7.3 对每个 loop 应用改写

`applyMultiBufferToGMLoadLoops()` 遍历所有 context，调用：

```cpp
applyMultiBufferToForLoop(context, allCtxForOps_)
```

`allCtxForOps_` 保存所有待改写 loop，用于 consumer body 克隆阶段跳过已经处理的嵌套 loop。

### 7.4 清理

`cleanupTransformedIR()`：

1. `deduplicateConstants(module)` 去重无 `ssbuffer.block_id` 的常量。
2. 删除被替换的原始 outer loop。
3. 对嵌套 loop 做保护：如果某个 context 的 forOp 位于另一个被处理 forOp 内，则不在这里重复 erase。

## 8. 依赖链计算

`DependencyAnalysis.cpp` 的核心是 `computeLoadChain(markedOp, scopeBlock)`。

它分四个 phase：

1. **Backward SSA trace**：从 marked load 反向追踪 operand 的 defining op。
2. **Forward trace from alloc**：如果链中包含 `memref.alloc`，沿 alloc result 的用户向前追踪，捕捉 subview/copy/to_tensor 等依赖。
3. **Backward trace from newly discovered ops**：对第二步新发现 op 再做反向追踪，补齐地址计算。
4. **Capture region free vars**：如果链上 op 包含 region，例如 `scf.if`，收集 region 内使用但定义在 scope block 的自由变量。

最后按 block 内顺序排序，形成可 clone 的拓扑顺序。

### 8.1 backing alloc

`findBackingAlloc()` 优先寻找 marked op 第一个 operand 的 defining op 是否是 `memref.alloc`。如果不是，则退化为链中第一个 alloc。

这个 alloc 是后续多 buffer 化的替换点：

```text
原 alloc result -> group.bufSlots[slotIdx][loadIdx]
```

### 8.2 consumer skip 分析

producer 侧需要重放 load 依赖链来填 buffer，但 consumer 侧不一定还需要保留整条链。

`computeSkipInConsumer()` 会判断链上 op 是否还被非链上 op 使用，或者是否被 loop yield 使用：

1. 如果链上 op 的 result 被外部使用，则该 op 必须在 consumer body 中保留。
2. 对这些必要 op 再做 operand 反向传播，保留其依赖。
3. 其余链上 op 可跳过，因为 producer 侧已经负责生成数据，consumer 侧只需要从 slot 选择 tensor。

这一步能减少新循环体中的重复地址计算和无用 op。

## 9. 循环改写核心

`applyMultiBufferToForLoop()` 是实际改写一个 `scf.for` 的主函数。

### 9.1 分配 buffer slot

`allocateBufferSlotsForGroups()` 对每个 group 调用 `allocateBufferSlots()`：

```text
group.bufSlots[slotIdx][loadIdx]
```

每个 slot 都是一个新的 `memref.alloc`，类型和属性来自原 backing alloc。

### 9.2 生成循环不变量

`emitLoopInvariantValues()` 在 loop 前生成：

```text
tripCount = ceildiv(upperBound - lowerBound, step)
```

它把 lower/upper/step cast 到 index 类型，用于 producerCounter 和 consumerCounter 的统一计算。

`emitGroupInvariantValues()` 为每个 group 生成：

```text
true flag
index 1
slot constants: 0, 1, ..., depth - 1
depth constant
```

这些值会被 producer/consumer 的条件和 counter 更新复用。

### 9.3 构建扩展版 scf.for

`buildExtendedFor()` 在原 loop 前新建一个 `scf.for`：

```text
原 iter_args
+ 每个 group:
  + flags[depth]
  + prodCounter
  + consCounter
```

同时建立 `IRMapping`：

1. 原 induction var -> 新 induction var。
2. 原 block iter_arg -> 新 block iter_arg。
3. 后续 clone 原 body 时继续复用该 mapping。

### 9.4 线性 iter_arg 投影

producer 需要“模拟未来某一轮循环”的地址生成，因此要把原循环变量映射到 producerCounter 对应的位置：

```text
projectedIV = lowerBound + producerCounter * step
```

对于原 loop 的 iter_arg，如果 yield 形式是：

```text
yield arith.addi(iterArg, loopInvariantDelta)
```

则 `collectLinearIterArgDeltas()` 会记录 delta。producer 侧可投影为：

```text
projectedIterArg = initArg + producerCounter * delta
```

不满足该线性形式的 iter_arg 会映射为新 loop 当前 block argument，不做跨迭代投影。

### 9.5 生成新循环体

`emitMultiBufferLoopBody()` 负责新 body：

1. 计算 consumer 侧需要跳过/保留的 chain op。
2. 找到每个 group 对应的 owner op，即 marked load 在旧 body 顶层 block 中的祖先。
3. 遍历旧 body：
   - 到达 owner op 前，插入 producer 填充和 slot 选择。
   - 按需 clone 原 op。
   - owner op 后，插入 slot release。
4. 对无法定位 owner 的 group，在 body 末尾补发 producer/release。
5. 生成组合后的 `scf.yield`。

## 10. Producer 逻辑

producer 对每个 slot 生成一个 `scf.if`：

```text
if flag[slot] == false && prodCounter < tripCount:
  projectedIV = lowerBound + prodCounter * step
  projectedIterArgs = init + prodCounter * delta
  clone address/load chain, but:
    original alloc -> slot buffer
    original marked load op itself不直接 clone
  flag[slot] = true
  prodCounter++
else:
  flag[slot], prodCounter 保持不变
```

对应函数：

| 函数 | 作用 |
| --- | --- |
| `createProducerFillCondition()` | 生成 `flag empty && prodCounter < tripCount`。 |
| `emitProducerFillForSlot()` | 生成一个 slot 的 producer `scf.if`。 |
| `mapProducerLoopPosition()` | 把旧 induction var 映射到 producerCounter 对应的逻辑迭代。 |
| `mapProducerIterArgs()` | 对线性 iter_arg 做投影。 |
| `mapProducerSlotBuffers()` | 把原 alloc result 映射到当前 slot buffer。 |
| `materializeProducerChain()` | clone load 的依赖链，完成真正填充。 |

producer 的核心价值是提前生成后续迭代的数据，把原来 consumer 路径中的 load 转移到独立的填充路径上。

## 11. Consumer 逻辑

consumer 根据 `consCounter % depth` 选择当前要消费的 slot：

```text
target = consCounter % depth
slotTensor[0] = to_tensor(slotBuffer[0])
slotTensor[1] = to_tensor(slotBuffer[1])
...
selected = select(target == 0, slotTensor[0],
           select(target == 1, slotTensor[1], ...))
```

对应函数：

| 函数 | 作用 |
| --- | --- |
| `emitConsumerSlotTarget()` | 计算 `target = consCounter % depth`。 |
| `emitLoadSlotSelection()` | 为每个 slot 生成 `bufferization.to_tensor`，再用 `cmp/select` 选择目标 tensor。 |
| `emitSlotFlagClear()` | 消费完成后，如果 `target == slotIdx`，把对应 flag 清为 false。 |
| `emitConsumerReleaseForGroup()` | 清 flag 并递增 `consCounter`。 |

原 marked load 的 result 会在 `IRMapping` 中映射到 `selected`，因此后续 clone 原 compute op 时会自然使用多 buffer 选出的 tensor。

## 12. IR 形态变化示意

### 12.1 改写前

```text
%r = scf.for %i = %lb to %ub step %step iter_args(...) -> (...) {
  %alloc = memref.alloc()
  %addr = ... %i ...
  %load_tensor = <gm load / to_tensor path> %alloc {gm_load_bufferable}
  %compute = ... %load_tensor ...
  scf.yield ...
}
```

### 12.2 改写后

```text
%slot0 = memref.alloc()
%slot1 = memref.alloc()
...

%new = scf.for %i = %lb to %ub step %step
  iter_args(
    original iter_args...,
    flag0 = false,
    flag1 = false,
    ...,
    prodCounter = 0,
    consCounter = 0
  ) {

  // producer: 填空 slot
  %flag0_next, %prod1 = scf.if (%flag0 == false && %prodCounter < %tripCount) {
    %projected_i = %lb + %prodCounter * %step
    clone address/load chain with %alloc -> %slot0
    scf.yield true, %prodCounter + 1
  } else {
    scf.yield %flag0, %prodCounter
  }

  // consumer: 选择当前 slot
  %target = %consCounter mod depth
  %tensor0 = bufferization.to_tensor %slot0
  %tensor1 = bufferization.to_tensor %slot1
  %selected = arith.select(%target == 0, %tensor0, %tensor1)

  // original compute, marked load result -> selected
  %compute = ... %selected ...

  // release
  %flag0_final = if (%target == 0) false else %flag0_next
  %consCounter_next = %consCounter + 1

  scf.yield original yields..., flags..., prodCounter..., consCounter_next
}
```

## 13. 清理与正确性保障

### 13.1 死 iter_arg 裁剪

新 loop 初始会保留原 loop 的 iter_args，再追加多 buffer 状态。改写后某些原 iter_arg 可能已经无外部使用，且其 body argument 只流向无副作用 op 和 yield。

`pruneDeadIterArgs()` 会：

1. 检查候选 iter_arg 是否 dead。
2. 创建只包含 live iter_args 的新 `scf.for`。
3. clone body 并重建 mapping。
4. 替换 live result 使用。
5. 删除旧 loop。

### 13.2 死 op 删除

`eraseDeadBodyOps()` 只删除：

1. 直接位于 loop body 的 op。
2. memory-effect-free。
3. 所有 result 都没有使用。

删除后会继续向 operand defining op 传播，清理链式死代码。

### 13.3 常量去重

`deduplicateConstants()` 对不带 `ssbuffer.block_id` 的 `arith.constant` 去重。带 block id 的常量不去重，因为 block id 是调试和阶段归属信息的一部分，错误合并会丢失归属语义。

### 13.4 嵌套 loop 防重复

`allCtxForOps_` 和 inner-first 排序共同保证：

1. 内层 loop 先改写。
2. 外层 clone body 时跳过已处理 loop。
3. cleanup 时避免对嵌套 context 重复 erase。

## 14. 适用条件与限制

该 pass 当前更适合以下场景：

1. load 在 `scf.for` 内，并且 loop trip count 大于配置的 buffer depth。
2. load 地址生成链与 block argument 或循环位置相关。
3. 地址生成可线性投影，尤其是 `reinterpret_cast` offset 不能依赖间接 load 或复杂控制流。
4. marked load 能找到 backing `memref.alloc`。
5. 原 iter_arg 若要在 producer 中准确投影，需要满足 `iterArg + loopInvariantDelta` 的线性更新形式。

会跳过或保守处理的情况：

1. `depth <= 1`：顶层 pass 直接返回。
2. 静态 trip count `<= depth`：对应 loop 不改写。
3. 只服务于 mask processing 的 load 不标记。
4. 非线性 `reinterpret_cast` offset 不标记。
5. 找不到 enclosing `scf.for` 或 backing alloc 的 marked op 不进入改写。

## 15. 调试入口

主要 debug type：

| Debug type | 文件 |
| --- | --- |
| `decouple-compute-and-memory` | `DecoupleComputeAndMemory.cpp` |
| `async-load-hoisting` | `AsyncLoadHoisting.cpp` |
| `add-multi-buffer-to-gm-load` | `AddMultiBufferToGMLoadInternal.h` 及相关 cpp |

常见观察点：

1. `AsyncLoadHoistingPass` 前后 IR：确认目标 load 是否带上 `gm_load_bufferable`。
2. `AddMultiBufferToGMLoadPass` 前后 IR：确认原 `scf.for` 是否被扩展，是否新增 `flags/prodCounter/consCounter`。
3. `ssbuffer.block_id`：确认新生成的 compare/select/if/constant 是否继承了合理 block id。
4. `logRepeatedTopLevelBlockRuns()`：检查改写后是否出现非连续的顶层 block id run。

## 16. 述职讲解建议

建议按下面顺序讲：

1. **背景问题**：循环内 GM load 与 compute 串行，访存延迟难隐藏。
2. **总体方案**：用多 buffer 把 load 变成 producer，把 compute 变成 consumer，利用 `depth` 个 slot 做环形流水。
3. **入口控制**：`BufferCountManager` 控制是否启用，`depth <= 1` 直接跳过，避免无收益改写。
4. **候选识别**：`AsyncLoadHoistingPass` 不是盲目改写，而是通过 read effect、block id、地址链线性检查筛出安全 load。
5. **依赖链构建**：`computeLoadChain()` 四阶段补齐 backward/forward/region free var，保证 producer 侧 clone 不缺 SSA 依赖。
6. **循环重写**：为每个 load group 分配 `depth` 个 slot，并扩展 `scf.for` 的 iter_args 存 flag 和 counter。
7. **producer 投影**：用 `producerCounter` 投影 induction var 和线性 iter_arg，模拟未来迭代的地址生成。
8. **consumer 替换**：用 `consCounter % depth` 选择 slot，把原 load result 映射到 selected tensor。
9. **收尾清理**：裁剪 dead iter_arg、删除死 op、常量去重、嵌套 loop 防重复。
10. **价值总结**：语义仍是按原迭代顺序消费数据，但数据准备被前移并流水化，提升访存计算重叠能力。

## 17. 可用于述职的简短总结

这段代码实现了一个 MLIR 层面的 GM load 多 buffer 优化。入口 pass 先根据配置的 LoadStore buffer depth 判断是否启用，然后串联候选 load 标记和循环重写两个 pass。候选识别阶段会分析 load 的 memory effect、`ssbuffer.block_id`、地址生成链以及 `reinterpret_cast` offset 是否线性可预测，避免对复杂或不安全的访存做改写。真正的改写阶段按最近的 `scf.for` 对 marked load 分组，为每个 load group 分配多份 memref slot，并把原 loop 扩展为带 slot flag、producer counter 和 consumer counter 的状态机。producer 根据 counter 投影未来迭代位置并重放 load 依赖链填充空 slot；consumer 根据 counter 选择当前 slot，把原 load result 替换为选中的 tensor，再执行原计算并释放 slot。最终通过 dead iter_arg 裁剪、死代码删除和常量去重保持 IR 简洁。

## 18. 后续维护关注点

1. 如果上游 IR 增加新的地址生成 op，需要确认 `AsyncLoadHoistingPass` 的线性检查是否仍然覆盖。
2. 如果 marked load 不再直接或间接由 `memref.alloc` 支撑，需要增强 `findBackingAlloc()`。
3. 如果要支持更复杂 iter_arg 更新，需要扩展 `getLinearIterArgDelta()` 的识别能力。
4. 如果一个 group 未来合并多个 load，需要重新审视当前“一 load 一 group”的 counter/flag 设计。
5. 新增测试时应覆盖：嵌套 loop、动态 trip count、非线性 offset、region free var、consumer 外部使用链上 op 等场景。
