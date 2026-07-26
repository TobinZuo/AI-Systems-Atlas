import {
  simulateRingAllReduce,
  type RankState,
  type RingRound,
  type RingSimulation,
  type RingTransfer,
} from "../sim/ring";

export type RingInputPresetId = "rank-pattern" | "same" | "mixed-sign";
export type RingTeachingPhase = "initial" | "reduce-scatter" | "all-gather" | "complete";

export interface RingTeachingStep {
  id: string;
  label: string;
  compactLabel: string;
  phase: RingTeachingPhase;
  round?: number;
}

export interface RingChunkHop {
  phase: "reduce-scatter" | "all-gather";
  round: number;
  from: number;
  to: number;
}

export interface RingCommunicationCost {
  worldSize: number;
  messageBytes: number;
  roundsPerPhase: number;
  totalRounds: number;
  chunkBytes: number;
  bytesPerRankPerPhase: number;
  bytesPerRankTotal: number;
  aggregateBytes: number;
  messageMultiplesPerRank: number;
}

const mod = (value: number, size: number) => ((value % size) + size) % size;

const cloneRankStates = (ranks: RankState[]): RankState[] =>
  ranks.map((rank) => ({
    rank: rank.rank,
    chunks: rank.chunks.map((chunk) => ({
      ...chunk,
      values: [...chunk.values],
      contributors: [...chunk.contributors],
    })),
  }));

function validateWorldSize(worldSize: number): void {
  if (!Number.isInteger(worldSize) || worldSize < 2 || worldSize > 8) {
    throw new Error("Teaching Ring world size must be an integer from 2 to 8");
  }
}

export function ringInputs(worldSize: number, preset: RingInputPresetId): number[][] {
  validateWorldSize(worldSize);

  return Array.from({ length: worldSize }, (_, rank) =>
    Array.from({ length: worldSize }, (_, chunk) => {
      if (preset === "same") return chunk + 1;
      if (preset === "mixed-sign") {
        const sign = rank % 2 === 0 ? 1 : -1;
        return sign * (rank + 1) * (chunk + 1);
      }
      return rank * 10 + chunk + 1;
    }),
  );
}

export function createRingTeachingSimulation(
  worldSize: number,
  preset: RingInputPresetId,
): RingSimulation {
  return simulateRingAllReduce(ringInputs(worldSize, preset));
}

export function ringTeachingSteps(worldSize: number): RingTeachingStep[] {
  validateWorldSize(worldSize);

  return [
    { id: "initial", label: "本地 Tensor 已切块", compactLabel: "Input", phase: "initial" },
    ...Array.from({ length: worldSize - 1 }, (_, round) => ({
      id: `rs-${round}`,
      label: `Reduce-Scatter 第 ${round + 1} 轮`,
      compactLabel: `RS ${round + 1}`,
      phase: "reduce-scatter" as const,
      round,
    })),
    ...Array.from({ length: worldSize - 1 }, (_, round) => ({
      id: `ag-${round}`,
      label: `All-Gather 第 ${round + 1} 轮`,
      compactLabel: `AG ${round + 1}`,
      phase: "all-gather" as const,
      round,
    })),
    { id: "complete", label: "每个 Rank 得到完整结果", compactLabel: "Done", phase: "complete" },
  ];
}

export function ringRoundForTeachingStep(
  simulation: RingSimulation,
  step: RingTeachingStep,
): RingRound | null {
  if (step.phase === "reduce-scatter") {
    return simulation.reduceScatter[step.round ?? 0] ?? null;
  }
  if (step.phase === "all-gather") {
    return simulation.allGather[step.round ?? 0] ?? null;
  }
  return null;
}

export function ringRanksForTeachingStep(
  simulation: RingSimulation,
  step: RingTeachingStep,
): RankState[] {
  if (step.phase === "initial") return cloneRankStates(simulation.initial);
  if (step.phase === "reduce-scatter") {
    const round = simulation.reduceScatter[step.round ?? 0];
    if (!round) throw new Error("Reduce-Scatter round is unavailable");
    return cloneRankStates(round.ranks);
  }
  if (step.phase === "all-gather") {
    const round = simulation.allGather[step.round ?? 0];
    if (!round) throw new Error("All-Gather round is unavailable");
    return cloneRankStates(round.ranks);
  }
  const finalRound = simulation.allGather[simulation.allGather.length - 1];
  if (!finalRound) throw new Error("Final All-Gather state is unavailable");
  return cloneRankStates(finalRound.ranks);
}

export function ringSendChunk(
  phase: "reduce-scatter" | "all-gather",
  rank: number,
  round: number,
  worldSize: number,
): number {
  validateWorldSize(worldSize);
  if (!Number.isInteger(rank) || rank < 0 || rank >= worldSize) {
    throw new Error("Rank is outside the Ring communicator");
  }
  if (!Number.isInteger(round) || round < 0 || round >= worldSize - 1) {
    throw new Error("Round is outside the Ring phase");
  }
  return phase === "reduce-scatter"
    ? mod(rank - round, worldSize)
    : mod(rank + 1 - round, worldSize);
}

export function ringOwnedChunkAfterReduce(rank: number, worldSize: number): number {
  validateWorldSize(worldSize);
  if (!Number.isInteger(rank) || rank < 0 || rank >= worldSize) {
    throw new Error("Rank is outside the Ring communicator");
  }
  return mod(rank + 1, worldSize);
}

export function ringTransferForRank(
  round: RingRound,
  rank: number,
): { outgoing: RingTransfer; incoming: RingTransfer } {
  const outgoing = round.transfers.find((transfer) => transfer.from === rank);
  const incoming = round.transfers.find((transfer) => transfer.to === rank);
  if (!outgoing || !incoming) throw new Error(`Round does not contain Rank ${rank}`);
  return { outgoing, incoming };
}

export function ringChunkJourney(
  simulation: RingSimulation,
  chunk: number,
): RingChunkHop[] {
  if (!Number.isInteger(chunk) || chunk < 0 || chunk >= simulation.worldSize) {
    throw new Error("Chunk is outside the Ring tensor");
  }

  return [...simulation.reduceScatter, ...simulation.allGather].map((round) => {
    const transfer = round.transfers.find((item) => item.chunk === chunk);
    if (!transfer) throw new Error(`Chunk ${chunk} is missing from Ring round`);
    return {
      phase: round.phase,
      round: round.round,
      from: transfer.from,
      to: transfer.to,
    };
  });
}

export function ringCommunicationCost(
  worldSize: number,
  messageBytes: number,
): RingCommunicationCost {
  validateWorldSize(worldSize);
  if (!Number.isFinite(messageBytes) || messageBytes <= 0) {
    throw new Error("Message bytes must be a positive finite number");
  }

  const roundsPerPhase = worldSize - 1;
  const totalRounds = roundsPerPhase * 2;
  const chunkBytes = messageBytes / worldSize;
  const bytesPerRankPerPhase = roundsPerPhase * chunkBytes;
  const bytesPerRankTotal = bytesPerRankPerPhase * 2;

  return {
    worldSize,
    messageBytes,
    roundsPerPhase,
    totalRounds,
    chunkBytes,
    bytesPerRankPerPhase,
    bytesPerRankTotal,
    aggregateBytes: bytesPerRankTotal * worldSize,
    messageMultiplesPerRank: bytesPerRankTotal / messageBytes,
  };
}
