export type DistributedStrategyId = "ddp" | "zero-1" | "fsdp";

export type ComparisonPhaseId =
  | "persistent"
  | "gradient-sync"
  | "optimizer-step"
  | "next-forward";

export type TrainingConstraintId =
  | "fits"
  | "optimizer-pressure"
  | "model-pressure";

export interface ModelPreset {
  id: "100m" | "1b" | "7b";
  label: string;
  parameterCount: number;
}

export interface StrategyMemorySnapshot {
  id: DistributedStrategyId;
  label: string;
  fullName: string;
  parameterBytes: number;
  gradientBytes: number;
  optimizerStateBytes: number;
  totalBytes: number;
  parameterFraction: number;
  gradientFraction: number;
  optimizerStateFraction: number;
  updateOwner: string;
}

export interface StrategyCommunicationSnapshot {
  id: DistributedStrategyId;
  gradientReceiveBytes: number;
  parameterReceiveBytes: number;
  totalReceiveBytes: number;
  sequence: string[];
  formula: string;
  qualification: string;
}

export interface StrategyPhaseInsight {
  id: DistributedStrategyId;
  label: string;
  title: string;
  explanation: string;
  dataState: string;
  communication: string;
}

export interface StrategyRecommendation {
  strategy: DistributedStrategyId;
  label: string;
  reason: string;
  tradeoff: string;
}

export const comparisonModelPresets: ModelPreset[] = [
  { id: "100m", label: "100M 参数", parameterCount: 100_000_000 },
  { id: "1b", label: "1B 参数", parameterCount: 1_000_000_000 },
  { id: "7b", label: "7B 参数", parameterCount: 7_000_000_000 },
];

const strategyLabels: Record<DistributedStrategyId, string> = {
  ddp: "DDP",
  "zero-1": "ZeRO-1",
  fsdp: "FSDP",
};

function validateInputs(parameterCount: number, worldSize: number): void {
  if (!Number.isFinite(parameterCount) || parameterCount <= 0) {
    throw new Error("parameterCount must be positive");
  }
  if (!Number.isInteger(worldSize) || worldSize < 1) {
    throw new Error("worldSize must be a positive integer");
  }
}

export function distributedMemorySnapshots(
  parameterCount: number,
  worldSize: number,
): StrategyMemorySnapshot[] {
  validateInputs(parameterCount, worldSize);
  const parameterBytes = parameterCount * 4;
  const shardedParameterBytes = parameterBytes / worldSize;

  return [
    {
      id: "ddp",
      label: "DDP",
      fullName: "Distributed Data Parallel",
      parameterBytes,
      gradientBytes: parameterBytes,
      optimizerStateBytes: parameterBytes * 2,
      totalBytes: parameterBytes * 4,
      parameterFraction: 1,
      gradientFraction: 1,
      optimizerStateFraction: 1,
      updateOwner: "每个 rank 都更新完整参数",
    },
    {
      id: "zero-1",
      label: "ZeRO-1",
      fullName: "Sharded Optimizer",
      parameterBytes,
      gradientBytes: parameterBytes,
      optimizerStateBytes: (parameterBytes * 2) / worldSize,
      totalBytes: parameterBytes * 2 + (parameterBytes * 2) / worldSize,
      parameterFraction: 1,
      gradientFraction: 1,
      optimizerStateFraction: 1 / worldSize,
      updateOwner: "每个参数只有一个 owner 更新",
    },
    {
      id: "fsdp",
      label: "FSDP",
      fullName: "Fully Sharded Data Parallel",
      parameterBytes: shardedParameterBytes,
      gradientBytes: shardedParameterBytes,
      optimizerStateBytes: shardedParameterBytes * 2,
      totalBytes: shardedParameterBytes * 4,
      parameterFraction: 1 / worldSize,
      gradientFraction: 1 / worldSize,
      optimizerStateFraction: 1 / worldSize,
      updateOwner: "每个 rank 只更新本地参数 shard",
    },
  ];
}

export function distributedCommunicationSnapshots(
  parameterCount: number,
  worldSize: number,
): StrategyCommunicationSnapshot[] {
  validateInputs(parameterCount, worldSize);
  const parameterBytes = parameterCount * 4;
  const collectiveFraction = (worldSize - 1) / worldSize;
  const allReduceReceiveBytes = 2 * collectiveFraction * parameterBytes;
  const oneWayCollectiveReceiveBytes = collectiveFraction * parameterBytes;

  return [
    {
      id: "ddp",
      gradientReceiveBytes: allReduceReceiveBytes,
      parameterReceiveBytes: 0,
      totalReceiveBytes: allReduceReceiveBytes,
      sequence: ["Backward 产生完整 dW", "Ring All-Reduce dW", "每个 rank 本地 AdamW"],
      formula: "2 × (N - 1) / N × P",
      qualification: "Ring All-Reduce 中每个 rank 接收的梯度 payload。",
    },
    {
      id: "zero-1",
      gradientReceiveBytes: allReduceReceiveBytes,
      parameterReceiveBytes: oneWayCollectiveReceiveBytes,
      totalReceiveBytes: allReduceReceiveBytes + oneWayCollectiveReceiveBytes,
      sequence: ["DDP All-Reduce dW", "Owner 本地 AdamW", "按参数顺序 Broadcast W"],
      formula: "2 × (N - 1) / N × P + (N - 1) / N × P",
      qualification: "前半是 Ring All-Reduce；后半是 owner 均匀时，每个 rank 平均接收的更新参数。",
    },
    {
      id: "fsdp",
      gradientReceiveBytes: oneWayCollectiveReceiveBytes,
      parameterReceiveBytes: oneWayCollectiveReceiveBytes * 2,
      totalReceiveBytes: oneWayCollectiveReceiveBytes * 3,
      sequence: ["Forward All-Gather W", "Backward All-Gather W", "Reduce-Scatter dW"],
      formula: "3 × (N - 1) / N × P",
      qualification: "按一轮训练中两次权重 All-Gather 和一次梯度 Reduce-Scatter 估算。",
    },
  ];
}

const phaseCopy: Record<ComparisonPhaseId, Record<DistributedStrategyId, Omit<StrategyPhaseInsight, "id" | "label">>> = {
  persistent: {
    ddp: {
      title: "四类状态都完整复制",
      explanation: "每张 GPU 长期保存完整参数、完整梯度和完整 AdamW m、v。",
      dataState: "W、dW、m、v 都是 P",
      communication: "此阶段没有集合通信",
    },
    "zero-1": {
      title: "只切 optimizer state",
      explanation: "参数和梯度仍完整复制；m、v 按参数 owner 分散到不同 rank。",
      dataState: "W=P，dW=P，m+v=2P/N",
      communication: "此阶段没有集合通信",
    },
    fsdp: {
      title: "四类状态都只留 shard",
      explanation: "完整模型平时不驻留在单卡，只保留本 rank 负责的参数、梯度和状态。",
      dataState: "W、dW、m、v 都约为 1/N",
      communication: "当前层完整 W 已释放",
    },
  },
  "gradient-sync": {
    ddp: {
      title: "完整梯度做 All-Reduce",
      explanation: "所有 rank 的 local dW 求和并平均，结果仍是每卡一份完整梯度。",
      dataState: "输入完整 local dW，输出完整 global dW",
      communication: "Reduce-Scatter + All-Gather",
    },
    "zero-1": {
      title: "梯度同步与 DDP 相同",
      explanation: "ZeRO-1 没有切梯度，因此 optimizer.step 前仍先得到每卡完整的平均梯度。",
      dataState: "输入完整 local dW，输出完整 global dW",
      communication: "Reduce-Scatter + All-Gather",
    },
    fsdp: {
      title: "完整 local dW 直接散成 shard",
      explanation: "Reduce-Scatter 一边跨 rank 归约，一边只把 owner 对应的梯度片段留下。",
      dataState: "输入临时完整 dW，输出平均 dW shard",
      communication: "Reduce-Scatter",
    },
  },
  "optimizer-step": {
    ddp: {
      title: "每个 rank 重复同一次更新",
      explanation: "参数、梯度、m、v 都相同，所以各 rank 独立执行 AdamW 后仍然一致。",
      dataState: "每卡更新完整 W、m、v",
      communication: "更新后无需同步参数",
    },
    "zero-1": {
      title: "Owner 更新，然后恢复副本一致",
      explanation: "每个 rank 只更新自己拥有的参数。短暂分歧后，owner 按相同顺序广播新参数。",
      dataState: "每卡只更新约 1/N 参数",
      communication: "逐参数 Broadcast W",
    },
    fsdp: {
      title: "本地 shard 就地更新",
      explanation: "参数 shard 与梯度 shard 已经对齐，AdamW 只更新本地 shard，不重建完整模型。",
      dataState: "每卡只更新约 1/N 参数",
      communication: "更新后无需 Broadcast",
    },
  },
  "next-forward": {
    ddp: {
      title: "完整参数已经在本地",
      explanation: "下一层 forward 直接读取本卡 HBM 中的完整权重。",
      dataState: "W 始终完整驻留",
      communication: "参数路径无通信",
    },
    "zero-1": {
      title: "广播完成后直接计算",
      explanation: "所有 owner 的新参数已经写回每张卡，因此下一轮 forward 不需要临时拼权重。",
      dataState: "W 再次完整且一致",
      communication: "参数路径无通信",
    },
    fsdp: {
      title: "算到一层才临时拼完整",
      explanation: "forward pre-hook All-Gather 当前层权重，Kernel 计算后立即 Reshard 并释放临时 buffer。",
      dataState: "W shard → 临时完整 W → W shard",
      communication: "每层 All-Gather W",
    },
  },
};

export function comparisonPhaseInsights(phase: ComparisonPhaseId): StrategyPhaseInsight[] {
  return (["ddp", "zero-1", "fsdp"] as DistributedStrategyId[]).map((id) => ({
    id,
    label: strategyLabels[id],
    ...phaseCopy[phase][id],
  }));
}

export function recommendDistributedStrategy(
  constraint: TrainingConstraintId,
): StrategyRecommendation {
  if (constraint === "fits") {
    return {
      strategy: "ddp",
      label: "先选 DDP",
      reason: "模型状态能放进单卡时，DDP 的状态与执行路径最直接。",
      tradeoff: "显存里会重复保存参数、梯度和 optimizer state。",
    };
  }
  if (constraint === "optimizer-pressure") {
    return {
      strategy: "zero-1",
      label: "先看 ZeRO-1",
      reason: "参数和梯度还能放下，只把 AdamW m、v 分片就能解除主要压力。",
      tradeoff: "owner 更新后需要额外同步参数，参数大小差异也可能造成分片不均。",
    };
  }
  return {
    strategy: "fsdp",
    label: "需要 FSDP",
    reason: "参数或梯度本身已经超过单卡预算，必须把三类模型状态一起分片。",
    tradeoff: "每层 forward 和 backward 都会引入权重 All-Gather，并增加临时 buffer 管理复杂度。",
  };
}

export function formatComparisonBytes(bytes: number): string {
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const digits = value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[unitIndex]}`;
}
