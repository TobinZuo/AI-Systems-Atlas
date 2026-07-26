import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  createRingTeachingSimulation,
  ringChunkJourney,
  ringCommunicationCost,
  ringInputs,
  ringOwnedChunkAfterReduce,
  ringSendChunk,
  ringTeachingSteps,
  ringTransferForRank,
} from "./ringAllReduce";

describe("Ring AllReduce teaching model", () => {
  it("generates one deterministic scalar per chunk and rank", () => {
    assert.deepEqual(ringInputs(4, "rank-pattern"), [
      [1, 2, 3, 4],
      [11, 12, 13, 14],
      [21, 22, 23, 24],
      [31, 32, 33, 34],
    ]);
    assert.deepEqual(createRingTeachingSimulation(4, "rank-pattern").reduced, [64, 68, 72, 76]);
  });

  it("uses a chunk permutation in every round so ranks never send the same chunk", () => {
    for (const worldSize of [2, 3, 4, 8]) {
      for (const phase of ["reduce-scatter", "all-gather"] as const) {
        for (let round = 0; round < worldSize - 1; round += 1) {
          const chunks = Array.from(
            { length: worldSize },
            (_, rank) => ringSendChunk(phase, rank, round, worldSize),
          );
          assert.equal(new Set(chunks).size, worldSize);
          assert.deepEqual([...chunks].sort((a, b) => a - b), Array.from({ length: worldSize }, (_, i) => i));
        }
      }
    }
  });

  it("matches the modular schedule with every simulated transfer", () => {
    const simulation = createRingTeachingSimulation(4, "rank-pattern");
    for (const round of [...simulation.reduceScatter, ...simulation.allGather]) {
      for (const transfer of round.transfers) {
        assert.equal(
          transfer.chunk,
          ringSendChunk(round.phase, transfer.from, round.round, simulation.worldSize),
        );
        assert.equal(transfer.to, (transfer.from + 1) % simulation.worldSize);
      }
    }
  });

  it("reduces on receive during Reduce-Scatter and only copies during All-Gather", () => {
    const simulation = createRingTeachingSimulation(4, "rank-pattern");
    const reduceTransfer = ringTransferForRank(simulation.reduceScatter[0], 1).incoming;
    assert.deepEqual(
      reduceTransfer.after,
      reduceTransfer.before!.map((value, index) => value + reduceTransfer.sent[index]),
    );

    const gatherTransfer = ringTransferForRank(simulation.allGather[0], 1).incoming;
    assert.equal(gatherTransfer.before, null);
    assert.deepEqual(gatherTransfer.after, gatherTransfer.sent);
  });

  it("leaves one complete reduced owner chunk before gathering every chunk", () => {
    const simulation = createRingTeachingSimulation(4, "rank-pattern");
    const reduced = simulation.reduceScatter[simulation.reduceScatter.length - 1];
    for (const rank of reduced.ranks) {
      const completed = rank.chunks.filter((chunk) => chunk.complete);
      assert.equal(completed.length, 1);
      assert.equal(completed[0].chunk, ringOwnedChunkAfterReduce(rank.rank, 4));
    }
    const gathered = simulation.allGather[simulation.allGather.length - 1];
    assert.ok(gathered.ranks.every((rank) => rank.chunks.every((chunk) => chunk.complete)));
  });

  it("tracks one chunk through exactly two times N minus one hops", () => {
    const simulation = createRingTeachingSimulation(8, "same");
    const journey = ringChunkJourney(simulation, 3);
    assert.equal(journey.length, 14);
    assert.equal(journey.filter((hop) => hop.phase === "reduce-scatter").length, 7);
    assert.equal(journey.filter((hop) => hop.phase === "all-gather").length, 7);
  });

  it("builds a phase rail that scales with world size", () => {
    assert.equal(ringTeachingSteps(2).length, 4);
    assert.equal(ringTeachingSteps(4).length, 8);
    assert.equal(ringTeachingSteps(8).length, 16);
  });

  it("computes textbook bandwidth volume per rank", () => {
    const cost = ringCommunicationCost(4, 1024);
    assert.equal(cost.roundsPerPhase, 3);
    assert.equal(cost.totalRounds, 6);
    assert.equal(cost.chunkBytes, 256);
    assert.equal(cost.bytesPerRankPerPhase, 768);
    assert.equal(cost.bytesPerRankTotal, 1536);
    assert.equal(cost.aggregateBytes, 6144);
    assert.equal(cost.messageMultiplesPerRank, 1.5);
  });

  it("rejects invalid teaching inputs", () => {
    assert.throws(() => ringInputs(1, "same"), /world size/);
    assert.throws(() => ringSendChunk("reduce-scatter", 4, 0, 4), /Rank/);
    assert.throws(() => ringSendChunk("all-gather", 0, 3, 4), /Round/);
    assert.throws(() => ringCommunicationCost(4, 0), /positive/);
  });
});
