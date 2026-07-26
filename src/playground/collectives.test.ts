import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  collectiveCallPacket,
  collectiveOperationDefinitions,
  defaultCollectiveConfig,
  rankViewAtPhase,
  simulateCollective,
  type CollectiveFault,
  type CollectiveOperation,
  type CollectiveReduction,
} from "./collectives";

describe("Collective communication teaching model", () => {
  it("broadcasts the root tensor to every rank", () => {
    const simulation = simulateCollective({ ...defaultCollectiveConfig, operation: "broadcast", rootRank: 2 });
    assert.ok(simulation.ranks.every((rank) => assert.deepEqual(rank.outputSlots[0].values, [21, 22]) === undefined));
  });

  it("reduces elementwise to only root or to every rank", () => {
    const reduce = simulateCollective({ ...defaultCollectiveConfig, operation: "reduce", reduction: "sum", rootRank: 1 });
    assert.deepEqual(reduce.ranks[1].outputSlots[0].values, [64, 68]);
    assert.ok(reduce.ranks.filter((rank) => rank.rank !== 1).every((rank) => rank.outputSlots[0].state === "not-applicable"));
    for (const reduction of ["sum", "avg", "max"] as CollectiveReduction[]) {
      const allReduce = simulateCollective({ ...defaultCollectiveConfig, operation: "all-reduce", reduction });
      const expected = reduction === "sum" ? [64, 68] : reduction === "avg" ? [16, 17] : [31, 32];
      assert.ok(allReduce.ranks.every((rank) => assert.deepEqual(rank.outputSlots[0].values, expected) === undefined));
    }
  });

  it("scatters root chunks and gathers rank-ordered chunks", () => {
    const scatter = simulateCollective({ ...defaultCollectiveConfig, operation: "scatter", rootRank: 1 });
    assert.deepEqual(scatter.ranks.map((rank) => rank.outputSlots[0].values), [[101, 102], [111, 112], [121, 122], [131, 132]]);
    const gather = simulateCollective({ ...defaultCollectiveConfig, operation: "gather", rootRank: 2 });
    assert.deepEqual(gather.ranks[2].outputSlots.map((slot) => slot.values), [[1, 2], [11, 12], [21, 22], [31, 32]]);
  });

  it("all-gathers rank-ordered chunks into every result buffer", () => {
    const simulation = simulateCollective(defaultCollectiveConfig);
    const expected = [[1, 2], [11, 12], [21, 22], [31, 32]];
    assert.ok(simulation.ranks.every((rank) => assert.deepEqual(rank.outputSlots.map((slot) => slot.values), expected) === undefined));
    const partial = rankViewAtPhase(simulation, 2, "transfer");
    assert.deepEqual(partial.outputSlots.map((slot) => slot.state), ["pending", "pending", "resident", "pending"]);
    assert.deepEqual(partial.outputSlots.map((slot) => slot.values), [null, null, [21, 22], null]);
  });

  it("reduce-scatters one reduced chunk per rank", () => {
    const simulation = simulateCollective({ ...defaultCollectiveConfig, operation: "reduce-scatter", reduction: "sum" });
    assert.deepEqual(simulation.ranks.map((rank) => rank.outputSlots[0].values), [[604, 608], [644, 648], [684, 688], [724, 728]]);
  });

  it("all-to-all transposes source and destination chunk axes", () => {
    const simulation = simulateCollective({ ...defaultCollectiveConfig, operation: "all-to-all" });
    assert.deepEqual(simulation.ranks[2].outputSlots.map((slot) => slot.values), [[21, 22], [121, 122], [221, 222], [321, 322]]);
  });

  it("passes pointer, count, dtype, operation, communicator, and stream to the backend", () => {
    const nccl = collectiveCallPacket(simulateCollective({ ...defaultCollectiveConfig, operation: "all-reduce" }), 1);
    assert.equal(nccl.sendPointer, nccl.receivePointer);
    assert.equal(nccl.count, 2);
    assert.equal(nccl.reduceOp, "ncclSum");
    assert.match(nccl.communicator, /rank=1/);
    assert.equal(nccl.stream, "CUDA comm stream");
    const gloo = collectiveCallPacket(simulateCollective({ ...defaultCollectiveConfig, backend: "gloo" }), 0);
    assert.equal(gloo.stream, "CPU worker thread");
  });

  it("blocks completion for missing, mismatched operation, count, or rooted contract", () => {
    for (const fault of ["missing-rank", "operation-mismatch", "count-mismatch"] as CollectiveFault[]) {
      const simulation = simulateCollective({ ...defaultCollectiveConfig, fault });
      assert.equal(simulation.fault.failurePhase, "contract");
      assert.ok(rankViewAtPhase(simulation, 0, "complete").outputSlots.every((slot) => slot.state === "pending"));
    }
    const rooted = simulateCollective({ ...defaultCollectiveConfig, operation: "broadcast", fault: "root-mismatch" });
    const symmetric = simulateCollective({ ...defaultCollectiveConfig, operation: "all-reduce", fault: "root-mismatch" });
    assert.equal(rooted.fault.failurePhase, "contract");
    assert.equal(symmetric.fault.failurePhase, null);
  });

  it("keeps all operation, phase, reduction, and backend states valid", () => {
    const operations = collectiveOperationDefinitions.map((definition) => definition.id);
    for (const operation of operations) for (const reduction of ["sum", "avg", "max"] as CollectiveReduction[]) {
      for (const backend of ["nccl", "gloo"] as const) {
        const simulation = simulateCollective({ operation, reduction, backend, rootRank: 3, fault: "none" });
        assert.equal(simulation.ranks.length, 4);
        for (let rank = 0; rank < 4; rank += 1) {
          for (const phase of ["inputs", "contract", "transfer", "complete"] as const) assert.equal(rankViewAtPhase(simulation, rank, phase).rank, rank);
          assert.ok(collectiveCallPacket(simulation, rank).count >= 1);
        }
      }
    }
  });

  it("rejects invalid roots and ranks", () => {
    assert.throws(() => simulateCollective({ ...defaultCollectiveConfig, rootRank: 4 }));
    const simulation = simulateCollective(defaultCollectiveConfig);
    assert.throws(() => rankViewAtPhase(simulation, 4, "complete"));
    assert.throws(() => collectiveCallPacket(simulation, -1));
  });
});
