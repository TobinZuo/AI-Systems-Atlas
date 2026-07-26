export type ComputeDtype = "fp32" | "fp16";

export interface FSDPLayer {
  id: string;
  name: string;
  shortName: string;
  rows: number;
  columns: number;
  baseValue: number;
  use: string;
}

export interface FSDPPhase {
  id: string;
  label: string;
  compactLabel: string;
  stage: "idle" | "all-gather" | "compute" | "reshard" | "reduce-scatter" | "optimizer";
  pass: "forward" | "backward" | "step";
  fullParameter: boolean;
  fullGradient: boolean;
  shardedGradient: boolean;
  explanation: string;
}

export interface ShardRow {
  paddedIndex: number;
  rowIndex: number | null;
  isPadding: boolean;
  value: number | null;
}

export interface FSDPRankState {
  rank: number;
  localRows: ShardRow[];
  visibleWeightRows: number[];
  gradientRows: Array<{ rowIndex: number; value: number }>;
  fullParameter: boolean;
  weightDtype: "fp32" | ComputeDtype;
}

export interface FSDPMemorySnapshot {
  persistentWeightBytes: number;
  persistentGradientBytes: number;
  optimizerStateBytes: number;
  transientFullWeightBytes: number;
  transientFullGradientBytes: number;
  totalBytes: number;
}

export const fsdpLayers: FSDPLayer[] = [
  { id: "embedding", name: "embedding.weight", shortName: "Embedding", rows: 10, columns: 4, baseValue: 100, use: "把 token id 映射到 4 维向量" },
  { id: "linear1", name: "linear1.weight", shortName: "Linear 1", rows: 8, columns: 4, baseValue: 200, use: "第一层特征变换" },
  { id: "lm-head", name: "lm_head.weight", shortName: "LM Head", rows: 10, columns: 4, baseValue: 300, use: "把隐藏状态映射回词表 logits" },
];

export const fsdpPhases: FSDPPhase[] = [
  {
    id: "sharded-idle",
    label: "长期分片状态",
    compactLabel: "Shard",
    stage: "idle",
    pass: "forward",
    fullParameter: false,
    fullGradient: false,
    shardedGradient: false,
    explanation: "每个 rank 只长期保存一段 FP32 master weight。完整参数此时不存在于任何单卡。",
  },
  {
    id: "forward-gather",
    label: "Forward 前 All-Gather",
    compactLabel: "AG W",
    stage: "all-gather",
    pass: "forward",
    fullParameter: true,
    fullGradient: false,
    shardedGradient: false,
    explanation: "所有 rank 交换本地行分片，按 rank 顺序拼出当前层完整权重。低精度模式会在通信前转换 dtype。",
  },
  {
    id: "forward-compute",
    label: "执行当前层 Forward",
    compactLabel: "Forward",
    stage: "compute",
    pass: "forward",
    fullParameter: true,
    fullGradient: false,
    shardedGradient: false,
    explanation: "当前层的 Kernel 使用临时完整权重计算输出。其他层仍保持分片，不会同时全部展开。",
  },
  {
    id: "forward-reshard",
    label: "Forward 后 Reshard",
    compactLabel: "Free W",
    stage: "reshard",
    pass: "forward",
    fullParameter: false,
    fullGradient: false,
    shardedGradient: false,
    explanation: "当前层算完后立即释放完整权重，parameter.data 重新指向本地 FP32 分片。",
  },
  {
    id: "backward-gather",
    label: "Backward 前再次 All-Gather",
    compactLabel: "AG W",
    stage: "all-gather",
    pass: "backward",
    fullParameter: true,
    fullGradient: false,
    shardedGradient: false,
    explanation: "反向计算 dX 仍需要完整 W，因此输出梯度 hook 会在当前层 backward 前再次重建权重。",
  },
  {
    id: "backward-compute",
    label: "计算完整参数梯度",
    compactLabel: "Backward",
    stage: "compute",
    pass: "backward",
    fullParameter: true,
    fullGradient: true,
    shardedGradient: false,
    explanation: "每个 rank 根据自己的 local batch 产生一份完整 dW。此刻各 rank 的梯度数值不同。",
  },
  {
    id: "gradient-reduce-scatter",
    label: "梯度 Reduce-Scatter",
    compactLabel: "RS dW",
    stage: "reduce-scatter",
    pass: "backward",
    fullParameter: false,
    fullGradient: false,
    shardedGradient: true,
    explanation: "完整梯度先除 world_size，再跨 rank 求和；每个 rank 只保留与本地权重同形状的平均梯度分片。",
  },
  {
    id: "local-optimizer",
    label: "Optimizer 更新本地分片",
    compactLabel: "Step",
    stage: "optimizer",
    pass: "step",
    fullParameter: false,
    fullGradient: false,
    shardedGradient: true,
    explanation: "optimizer 只看见 FP32 权重分片和同形状梯度分片，更新后不需要再广播完整参数。",
  },
];

export function rowsPerShard(layer: FSDPLayer, worldSize: number): number {
  if (!Number.isInteger(worldSize) || worldSize < 1) throw new Error("worldSize must be positive");
  return Math.ceil(layer.rows / worldSize);
}

export function paddedRows(layer: FSDPLayer, worldSize: number): number {
  return rowsPerShard(layer, worldSize) * worldSize;
}

export function shardRowsForRank(
  layer: FSDPLayer,
  worldSize: number,
  rank: number,
): ShardRow[] {
  const shardRows = rowsPerShard(layer, worldSize);
  const start = rank * shardRows;
  return Array.from({ length: shardRows }, (_, offset) => {
    const paddedIndex = start + offset;
    const isPadding = paddedIndex >= layer.rows;
    return {
      paddedIndex,
      rowIndex: isPadding ? null : paddedIndex,
      isPadding,
      value: isPadding ? null : layer.baseValue + paddedIndex,
    };
  });
}

export function localGradientContribution(rank: number, rowIndex: number): number {
  return (rank + 1) * 10 + rowIndex + 1;
}

export function averagedGradient(worldSize: number, rowIndex: number): number {
  let sum = 0;
  for (let rank = 0; rank < worldSize; rank += 1) {
    sum += localGradientContribution(rank, rowIndex);
  }
  return sum / worldSize;
}

export function fsdpRankStates(
  layer: FSDPLayer,
  worldSize: number,
  phase: FSDPPhase,
  computeDtype: ComputeDtype,
): FSDPRankState[] {
  return Array.from({ length: worldSize }, (_, rank) => {
    const localRows = shardRowsForRank(layer, worldSize, rank);
    const actualLocalRows = localRows.filter((row) => !row.isPadding) as Array<ShardRow & { rowIndex: number }>;
    const visibleWeightRows = phase.fullParameter
      ? Array.from({ length: layer.rows }, (__, rowIndex) => rowIndex)
      : actualLocalRows.map((row) => row.rowIndex);

    let gradientRows: Array<{ rowIndex: number; value: number }> = [];
    if (phase.fullGradient) {
      gradientRows = Array.from({ length: layer.rows }, (__, rowIndex) => ({
        rowIndex,
        value: localGradientContribution(rank, rowIndex),
      }));
    } else if (phase.shardedGradient) {
      gradientRows = actualLocalRows.map((row) => ({
        rowIndex: row.rowIndex,
        value: averagedGradient(worldSize, row.rowIndex),
      }));
    }

    return {
      rank,
      localRows,
      visibleWeightRows,
      gradientRows,
      fullParameter: phase.fullParameter,
      weightDtype: phase.fullParameter ? computeDtype : "fp32",
    };
  });
}

export function fsdpMemorySnapshot(
  layer: FSDPLayer,
  worldSize: number,
  phase: FSDPPhase,
  computeDtype: ComputeDtype,
): FSDPMemorySnapshot {
  const fp32Bytes = 4;
  const computeBytes = computeDtype === "fp16" ? 2 : 4;
  const shardElements = rowsPerShard(layer, worldSize) * layer.columns;
  const fullElements = layer.rows * layer.columns;
  const persistentWeightBytes = shardElements * fp32Bytes;
  const persistentGradientBytes = phase.shardedGradient ? shardElements * fp32Bytes : 0;
  const optimizerStateBytes = shardElements * fp32Bytes * 2;
  const transientFullWeightBytes = phase.fullParameter ? fullElements * computeBytes : 0;
  const transientFullGradientBytes = phase.fullGradient ? fullElements * computeBytes : 0;

  return {
    persistentWeightBytes,
    persistentGradientBytes,
    optimizerStateBytes,
    transientFullWeightBytes,
    transientFullGradientBytes,
    totalBytes: persistentWeightBytes + persistentGradientBytes + optimizerStateBytes + transientFullWeightBytes + transientFullGradientBytes,
  };
}

export function allGatherPayloadBytes(
  layer: FSDPLayer,
  worldSize: number,
  computeDtype: ComputeDtype,
): number {
  const bytesPerElement = computeDtype === "fp16" ? 2 : 4;
  return rowsPerShard(layer, worldSize) * layer.columns * bytesPerElement;
}
