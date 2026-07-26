import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  allGatherPayloadBytes,
  averagedGradient,
  fsdpLayers,
  fsdpMemorySnapshot,
  fsdpPhases,
  fsdpRankStates,
  paddedRows,
  rowsPerShard,
  shardRowsForRank,
} from "./fsdp";

describe("FSDP parameter-lifecycle teaching model", () => {
  const paddedLayer = fsdpLayers[0];

  it("creates equal-sized shards and pads rows that do not divide evenly", () => {
    assert.equal(rowsPerShard(paddedLayer, 4), 3);
    assert.equal(paddedRows(paddedLayer, 4), 12);

    const rows = Array.from({ length: 4 }, (_, rank) => shardRowsForRank(paddedLayer, 4, rank));
    assert.ok(rows.every((shard) => shard.length === 3));
    assert.equal(rows.flat().filter((row) => row.isPadding).length, 2);
    assert.deepEqual(
      rows.flat().filter((row) => !row.isPadding).map((row) => row.rowIndex),
      Array.from({ length: 10 }, (_, index) => index),
    );
  });

  it("temporarily exposes full rows only during unsharded phases", () => {
    const gather = fsdpPhases.find((phase) => phase.id === "forward-gather")!;
    const reshard = fsdpPhases.find((phase) => phase.id === "forward-reshard")!;
    assert.ok(fsdpRankStates(paddedLayer, 2, gather, "fp32").every((rank) => rank.visibleWeightRows.length === 10));
    assert.ok(fsdpRankStates(paddedLayer, 2, reshard, "fp32").every((rank) => rank.visibleWeightRows.length === 5));
  });

  it("reduce-scatters the averaged gradient to matching local rows", () => {
    const reduceScatter = fsdpPhases.find((phase) => phase.id === "gradient-reduce-scatter")!;
    const states = fsdpRankStates(paddedLayer, 2, reduceScatter, "fp32");
    for (const state of states) {
      assert.equal(state.gradientRows.length, 5);
      for (const row of state.gradientRows) {
        assert.equal(row.value, averagedGradient(2, row.rowIndex));
        assert.ok(state.visibleWeightRows.includes(row.rowIndex));
      }
    }
  });

  it("uses low precision for communication while keeping the master shard fp32", () => {
    assert.equal(allGatherPayloadBytes(paddedLayer, 2, "fp16") * 2, allGatherPayloadBytes(paddedLayer, 2, "fp32"));
    const gather = fsdpPhases.find((phase) => phase.id === "forward-gather")!;
    const fp16 = fsdpMemorySnapshot(paddedLayer, 2, gather, "fp16");
    const fp32 = fsdpMemorySnapshot(paddedLayer, 2, gather, "fp32");
    assert.equal(fp16.persistentWeightBytes, fp32.persistentWeightBytes);
    assert.ok(fp16.transientFullWeightBytes < fp32.transientFullWeightBytes);
  });

  it("accounts for the temporary full gradient before Reduce-Scatter", () => {
    const backward = fsdpPhases.find((phase) => phase.id === "backward-compute")!;
    const reduceScatter = fsdpPhases.find((phase) => phase.id === "gradient-reduce-scatter")!;
    const before = fsdpMemorySnapshot(paddedLayer, 2, backward, "fp32");
    const after = fsdpMemorySnapshot(paddedLayer, 2, reduceScatter, "fp32");
    assert.ok(before.transientFullGradientBytes > 0);
    assert.equal(after.transientFullGradientBytes, 0);
    assert.ok(after.persistentGradientBytes > 0);
  });
});
