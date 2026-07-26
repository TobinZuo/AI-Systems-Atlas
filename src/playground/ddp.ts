import type { RankState, RingRound, RingSimulation } from "../sim/ring";

export type PlaygroundStage =
  | "local"
  | "reduce-scatter"
  | "all-gather"
  | "average"
  | "optimizer";

export interface PlaygroundStep {
  id: string;
  label: string;
  compactLabel: string;
  stage: PlaygroundStage;
  round?: number;
  conceptEventId: string;
}

export interface GradientPreset {
  id: string;
  label: string;
  description: string;
  values: number[][];
}

export const playgroundSteps: PlaygroundStep[] = [
  {
    id: "local",
    label: "本地梯度",
    compactLabel: "Local",
    stage: "local",
    conceptEventId: "gradient-writeback",
  },
  ...Array.from({ length: 3 }, (_, round) => ({
    id: `rs-${round}`,
    label: `归约第 ${round + 1} 轮`,
    compactLabel: `RS ${round + 1}`,
    stage: "reduce-scatter" as const,
    round,
    conceptEventId: `reduce-scatter-${round}`,
  })),
  ...Array.from({ length: 3 }, (_, round) => ({
    id: `ag-${round}`,
    label: `收集第 ${round + 1} 轮`,
    compactLabel: `AG ${round + 1}`,
    stage: "all-gather" as const,
    round,
    conceptEventId: `all-gather-${round}`,
  })),
  {
    id: "average",
    label: "得到平均梯度",
    compactLabel: "AVG",
    stage: "average",
    conceptEventId: "bucket-writeback",
  },
  {
    id: "optimizer",
    label: "AdamW 更新",
    compactLabel: "AdamW",
    stage: "optimizer",
    conceptEventId: "optimizer-update",
  },
];

export const gradientPresets: GradientPreset[] = [
  {
    id: "place-values",
    label: "位值例子",
    description: "每个 rank 相差一个数量级，方便追踪来源。",
    values: [
      [1, 2, 3, 4, 5, 6, 7, 8],
      [10, 20, 30, 40, 50, 60, 70, 80],
      [100, 200, 300, 400, 500, 600, 700, 800],
      [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
    ],
  },
  {
    id: "same",
    label: "完全相同",
    description: "四张 GPU 给出相同梯度，最终平均值不变。",
    values: Array.from({ length: 4 }, () => [1, 2, 3, 4, 5, 6, 7, 8]),
  },
  {
    id: "mixed-sign",
    label: "正负抵消",
    description: "观察不同 rank 的正负梯度如何在归约时抵消。",
    values: [
      [4, 3, 2, 1, -1, -2, -3, -4],
      [-1, -2, -3, -4, 4, 3, 2, 1],
      [2, 2, 2, 2, -2, -2, -2, -2],
      [-3, -1, 1, 3, 3, 1, -1, -3],
    ],
  },
  {
    id: "large-rank",
    label: "异常 Rank",
    description: "Rank 3 的梯度更大，看看它如何影响全局平均值。",
    values: [
      [1, 1, 1, 1, 1, 1, 1, 1],
      [2, 2, 2, 2, 2, 2, 2, 2],
      [3, 3, 3, 3, 3, 3, 3, 3],
      [20, 20, 20, 20, 20, 20, 20, 20],
    ],
  },
];

const cloneRanks = (ranks: RankState[]): RankState[] =>
  ranks.map((rank) => ({
    rank: rank.rank,
    chunks: rank.chunks.map((chunk) => ({
      ...chunk,
      values: [...chunk.values],
      contributors: [...chunk.contributors],
    })),
  }));

export function ranksForStep(
  simulation: RingSimulation,
  step: PlaygroundStep,
): RankState[] {
  if (step.stage === "local") return cloneRanks(simulation.initial);
  if (step.stage === "reduce-scatter") {
    return cloneRanks(simulation.reduceScatter[step.round ?? 0].ranks);
  }
  if (step.stage === "all-gather") {
    return cloneRanks(simulation.allGather[step.round ?? 0].ranks);
  }

  const gathered = cloneRanks(
    simulation.allGather[simulation.allGather.length - 1].ranks,
  );
  if (step.stage === "average" || step.stage === "optimizer") {
    for (const rank of gathered) {
      for (const chunk of rank.chunks) {
        chunk.values = chunk.values.map(
          (value) => value / simulation.worldSize,
        );
      }
    }
  }
  return gathered;
}

export function roundForStep(
  simulation: RingSimulation,
  step: PlaygroundStep,
): RingRound | null {
  if (step.stage === "reduce-scatter") {
    return simulation.reduceScatter[step.round ?? 0];
  }
  if (step.stage === "all-gather") {
    return simulation.allGather[step.round ?? 0];
  }
  return null;
}

export function stepIndexForConceptEvent(eventId: string): number {
  const exact = playgroundSteps.findIndex(
    (step) => step.conceptEventId === eventId,
  );
  if (exact >= 0) return exact;
  if (eventId === "python-backward" || eventId === "autograd-schedules") return 0;
  if (eventId === "gradient-kernel" || eventId === "bucket-ready") return 0;
  if (eventId === "all-gather-2") return 6;
  if (eventId === "iteration-complete") return playgroundSteps.length - 1;
  return 0;
}

export function copyPreset(values: number[][]): number[][] {
  return values.map((rank) => [...rank]);
}
