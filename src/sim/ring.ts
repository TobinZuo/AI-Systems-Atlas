export type Values = readonly number[];

export interface ChunkState {
  chunk: number;
  values: number[];
  contributors: number[];
  complete: boolean;
}

export interface RankState {
  rank: number;
  chunks: ChunkState[];
}

export interface RingTransfer {
  from: number;
  to: number;
  chunk: number;
  sent: number[];
  before: number[] | null;
  after: number[];
  contributors: number[];
}

export interface RingRound {
  phase: "reduce-scatter" | "all-gather";
  round: number;
  transfers: RingTransfer[];
  ranks: RankState[];
}

export interface RingSimulation {
  worldSize: number;
  chunkSize: number;
  inputs: number[][];
  initial: RankState[];
  reduceScatter: RingRound[];
  allGather: RingRound[];
  reduced: number[];
  averaged: number[];
}

const mod = (value: number, size: number) => ((value % size) + size) % size;

const add = (left: Values, right: Values) =>
  left.map((value, index) => value + right[index]);

const cloneRanks = (ranks: RankState[]): RankState[] =>
  ranks.map((rank) => ({
    rank: rank.rank,
    chunks: rank.chunks.map((chunk) => ({
      ...chunk,
      values: [...chunk.values],
      contributors: [...chunk.contributors],
    })),
  }));

/**
 * Simulate the textbook unidirectional Ring All-Reduce.
 *
 * Reduce-Scatter round s:
 *   rank r sends chunk (r - s) mod N to rank (r + 1) mod N.
 * All-Gather starts with the one completed chunk owned by each rank and sends
 * the most recently received completed chunk to the next rank.
 */
export function simulateRingAllReduce(inputs: number[][]): RingSimulation {
  const worldSize = inputs.length;
  if (worldSize < 2) throw new Error("Ring All-Reduce needs at least two ranks");
  const width = inputs[0]?.length ?? 0;
  if (!width || inputs.some((values) => values.length !== width)) {
    throw new Error("Every rank must provide a non-empty tensor of equal length");
  }
  if (width % worldSize !== 0) {
    throw new Error("Tensor width must be divisible by world size");
  }

  const chunkSize = width / worldSize;
  const initial: RankState[] = inputs.map((values, rank) => ({
    rank,
    chunks: Array.from({ length: worldSize }, (_, chunk) => ({
      chunk,
      values: values.slice(chunk * chunkSize, (chunk + 1) * chunkSize),
      contributors: [rank],
      complete: false,
    })),
  }));

  let reduceState = cloneRanks(initial);
  const reduceScatter: RingRound[] = [];
  for (let round = 0; round < worldSize - 1; round += 1) {
    const nextState = cloneRanks(reduceState);
    const transfers: RingTransfer[] = [];

    for (let from = 0; from < worldSize; from += 1) {
      const to = mod(from + 1, worldSize);
      const chunk = mod(from - round, worldSize);
      const outgoing = reduceState[from].chunks[chunk];
      const destination = reduceState[to].chunks[chunk];
      const after = add(destination.values, outgoing.values);
      const contributors = Array.from(
        new Set([...destination.contributors, ...outgoing.contributors]),
      ).sort((a, b) => a - b);

      nextState[to].chunks[chunk] = {
        chunk,
        values: after,
        contributors,
        complete: contributors.length === worldSize,
      };
      transfers.push({
        from,
        to,
        chunk,
        sent: [...outgoing.values],
        before: [...destination.values],
        after,
        contributors,
      });
    }

    reduceState = nextState;
    reduceScatter.push({
      phase: "reduce-scatter",
      round,
      transfers,
      ranks: cloneRanks(reduceState),
    });
  }

  const gatheredState: RankState[] = Array.from(
    { length: worldSize },
    (_, rank) => ({
      rank,
      chunks: Array.from({ length: worldSize }, (_, chunk) => {
        const ownedChunk = mod(rank + 1, worldSize);
        const completed = reduceState[rank].chunks[ownedChunk];
        return chunk === ownedChunk
          ? { ...completed, values: [...completed.values], complete: true }
          : { chunk, values: [], contributors: [], complete: false };
      }),
    }),
  );

  let gatherState = gatheredState;
  const allGather: RingRound[] = [];
  for (let round = 0; round < worldSize - 1; round += 1) {
    const nextState = cloneRanks(gatherState);
    const transfers: RingTransfer[] = [];

    for (let from = 0; from < worldSize; from += 1) {
      const to = mod(from + 1, worldSize);
      const chunk = mod(from + 1 - round, worldSize);
      const outgoing = gatherState[from].chunks[chunk];
      if (!outgoing.complete) {
        throw new Error(`Rank ${from} does not own completed chunk ${chunk}`);
      }
      nextState[to].chunks[chunk] = {
        ...outgoing,
        values: [...outgoing.values],
        contributors: [...outgoing.contributors],
      };
      transfers.push({
        from,
        to,
        chunk,
        sent: [...outgoing.values],
        before: null,
        after: [...outgoing.values],
        contributors: [...outgoing.contributors],
      });
    }

    gatherState = nextState;
    allGather.push({
      phase: "all-gather",
      round,
      transfers,
      ranks: cloneRanks(gatherState),
    });
  }

  const reduced = Array.from({ length: width }, (_, index) =>
    inputs.reduce((sum, rank) => sum + rank[index], 0),
  );

  return {
    worldSize,
    chunkSize,
    inputs: inputs.map((values) => [...values]),
    initial,
    reduceScatter,
    allGather,
    reduced,
    averaged: reduced.map((value) => value / worldSize),
  };
}

export const ddpGradientExample = simulateRingAllReduce([
  [1, 2, 3, 4, 5, 6, 7, 8],
  [10, 20, 30, 40, 50, 60, 70, 80],
  [100, 200, 300, 400, 500, 600, 700, 800],
  [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
]);
