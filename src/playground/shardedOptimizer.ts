export interface OptimizerGroup {
  id: string;
  label: string;
  learningRate: number;
  weightDecay: number;
}

export interface ShardedOptimizerParameter {
  id: string;
  index: number;
  name: string;
  shortName: string;
  groupId: string;
  elements: number;
  value: number;
  gradient: number;
  note?: string;
}

export interface AdamWUpdate {
  expAvg: number;
  expAvgSq: number;
  correctedAvg: number;
  correctedAvgSq: number;
  nextValue: number;
}

export interface ShardedOptimizerPhase {
  id: string;
  label: string;
  compactLabel: string;
  kind: "ready" | "owner-update" | "broadcast" | "complete";
  broadcastParameterIndex?: number;
  explanation: string;
}

export interface RankOptimizerMemory {
  rank: number;
  parameterBytes: number;
  gradientBytes: number;
  optimizerStateBytes: number;
  totalBytes: number;
  ownedParameterIds: string[];
}

export const optimizerGroups: OptimizerGroup[] = [
  { id: "fast", label: "参数组 A", learningRate: 0.02, weightDecay: 0 },
  { id: "decay", label: "参数组 B", learningRate: 0.01, weightDecay: 0.05 },
];

export const shardedOptimizerParameters: ShardedOptimizerParameter[] = [
  { id: "first-weight", index: 0, name: "first.weight", shortName: "W0", groupId: "fast", elements: 64, value: 0.5, gradient: 0.2 },
  { id: "first-bias", index: 1, name: "first.bias", shortName: "b0", groupId: "fast", elements: 8, value: 0.1, gradient: -0.1 },
  { id: "norm-weight", index: 2, name: "norm.weight", shortName: "γ", groupId: "decay", elements: 8, value: 1, gradient: 0.05 },
  { id: "tied-weight", index: 3, name: "middle/last.weight", shortName: "W1", groupId: "decay", elements: 64, value: 0.8, gradient: 0.12, note: "两个层共享同一个 Parameter，只分配一次 owner" },
  { id: "output-bias", index: 4, name: "output.bias", shortName: "b1", groupId: "decay", elements: 8, value: 0, gradient: -0.08 },
  { id: "head-weight", index: 5, name: "lm_head.weight", shortName: "W2", groupId: "decay", elements: 64, value: 0.3, gradient: 0.14 },
];

export const shardedOptimizerPhases: ShardedOptimizerPhase[] = [
  {
    id: "ready",
    label: "DDP 已同步梯度",
    compactLabel: "Grad ready",
    kind: "ready",
    explanation: "每个 rank 仍有完整参数和完整梯度，但 AdamW 状态只放在参数 owner 上。",
  },
  {
    id: "owner-update",
    label: "Owner 本地更新",
    compactLabel: "Local step",
    kind: "owner-update",
    explanation: "每个 rank 的本地 AdamW 只更新自己拥有的参数。此刻不同 rank 的完整模型暂时不一致。",
  },
  ...shardedOptimizerParameters.map((parameter) => ({
    id: `broadcast-${parameter.id}`,
    label: `广播 ${parameter.shortName}`,
    compactLabel: `Bcast ${parameter.shortName}`,
    kind: "broadcast" as const,
    broadcastParameterIndex: parameter.index,
    explanation: `所有 rank 同时进入同一个 broadcast，由 owner = parameter_index % world_size 对应的 rank 提供 ${parameter.name} 新值。`,
  })),
  {
    id: "complete",
    label: "模型副本恢复一致",
    compactLabel: "Consistent",
    kind: "complete",
    explanation: "全部参数按相同顺序广播完成。每个 rank 再次持有相同模型，可以进入下一轮 forward。",
  },
];

export function ownerForParameter(parameterIndex: number, worldSize: number): number {
  if (!Number.isInteger(worldSize) || worldSize < 1) throw new Error("worldSize must be positive");
  return parameterIndex % worldSize;
}

export function adamWFirstStep(
  parameter: ShardedOptimizerParameter,
  beta1 = 0.9,
  beta2 = 0.999,
  epsilon = 1e-8,
): AdamWUpdate {
  const group = optimizerGroups.find((item) => item.id === parameter.groupId);
  if (!group) throw new Error(`Missing optimizer group ${parameter.groupId}`);

  const expAvg = (1 - beta1) * parameter.gradient;
  const expAvgSq = (1 - beta2) * parameter.gradient ** 2;
  const correctedAvg = expAvg / (1 - beta1);
  const correctedAvgSq = expAvgSq / (1 - beta2);
  const normalized = correctedAvg / (Math.sqrt(correctedAvgSq) + epsilon);
  const nextValue = parameter.value
    - group.learningRate * normalized
    - group.learningRate * group.weightDecay * parameter.value;

  return { expAvg, expAvgSq, correctedAvg, correctedAvgSq, nextValue };
}

export function parameterValueAtPhase(
  parameter: ShardedOptimizerParameter,
  rank: number,
  worldSize: number,
  phase: ShardedOptimizerPhase,
): number {
  const owner = ownerForParameter(parameter.index, worldSize);
  const updated = adamWFirstStep(parameter).nextValue;

  if (phase.kind === "ready") return parameter.value;
  if (phase.kind === "complete") return updated;
  if (phase.kind === "owner-update") return rank === owner ? updated : parameter.value;

  const hasBroadcast = parameter.index <= (phase.broadcastParameterIndex ?? -1);
  return hasBroadcast || rank === owner ? updated : parameter.value;
}

export function optimizerMemoryByRank(worldSize: number): RankOptimizerMemory[] {
  const fp32Bytes = 4;
  const totalElements = shardedOptimizerParameters.reduce((sum, parameter) => sum + parameter.elements, 0);
  const parameterBytes = totalElements * fp32Bytes;
  const gradientBytes = parameterBytes;

  return Array.from({ length: worldSize }, (_, rank) => {
    const owned = shardedOptimizerParameters.filter(
      (parameter) => ownerForParameter(parameter.index, worldSize) === rank,
    );
    const optimizerStateBytes = owned.reduce(
      (sum, parameter) => sum + parameter.elements * fp32Bytes * 2,
      0,
    );
    return {
      rank,
      parameterBytes,
      gradientBytes,
      optimizerStateBytes,
      totalBytes: parameterBytes + gradientBytes + optimizerStateBytes,
      ownedParameterIds: owned.map((parameter) => parameter.id),
    };
  });
}

export function ddpOptimizerMemoryBytes(): number {
  const totalElements = shardedOptimizerParameters.reduce((sum, parameter) => sum + parameter.elements, 0);
  return totalElements * 4 * 4;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(2)} KiB`;
}
