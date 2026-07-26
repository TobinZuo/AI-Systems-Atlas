import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { fsdpPhases } from "./fsdp";
import { shardedOptimizerPhases } from "./shardedOptimizer";
import {
  comparisonHardwareSnapshot,
  distributedHardwareDataFlow,
  distributedHardwareRankViews,
  fsdpHardwareSnapshot,
  shardedOptimizerHardwareSnapshot,
} from "./distributedHardware";

describe("distributed hardware-path teaching model", () => {
  it("runs ZeRO owner updates on compute and broadcasts on comm", () => {
    const update = shardedOptimizerHardwareSnapshot(
      shardedOptimizerPhases.find((phase) => phase.kind === "owner-update")!,
      "weight",
      0,
    );
    const broadcast = shardedOptimizerHardwareSnapshot(
      shardedOptimizerPhases.find((phase) => phase.kind === "broadcast")!,
      "weight",
      0,
    );
    assert.equal(update.activeStream, "compute");
    assert.equal(update.link, "无链路传输");
    assert.equal(broadcast.activeStream, "comm");
    assert.equal(broadcast.link, "NVLink / PCIe P2P");
    assert.equal(broadcast.pattern, "broadcast");
    assert.deepEqual(
      distributedHardwareRankViews(broadcast, 4, 1).map((rank) => rank.role),
      ["source", "receiver", "receiver", "receiver"],
    );
    assert.deepEqual(distributedHardwareDataFlow(broadcast, 4).sourceRanks, [0]);
    assert.deepEqual(distributedHardwareDataFlow(broadcast, 4).destinationRanks, [1, 2, 3]);
    assert.match(distributedHardwareDataFlow(broadcast, 4).sourceObject, /updated weight/);
  });

  it("shows FSDP weight gathers and gradient reduce-scatter on the comm stream", () => {
    const gather = fsdpHardwareSnapshot(
      fsdpPhases.find((phase) => phase.stage === "all-gather")!,
      "linear.weight",
      "fp16",
    );
    const reduceScatter = fsdpHardwareSnapshot(
      fsdpPhases.find((phase) => phase.stage === "reduce-scatter")!,
      "linear.weight",
      "fp16",
    );
    assert.equal(gather.activeStream, "comm");
    assert.match(gather.kernel, /All-Gather/);
    assert.equal(gather.pattern, "all-gather");
    assert.ok(distributedHardwareRankViews(gather, 4, 2).every((rank) => rank.role === "exchange"));
    assert.ok(
      distributedHardwareRankViews(gather, 4, 2).every((rank) =>
        rank.memorySlots.some((slot) => slot.label === "W shard" && slot.state === "send")
        && rank.memorySlots.some((slot) => slot.label === "full W" && slot.state === "receive")),
    );
    assert.deepEqual(distributedHardwareDataFlow(gather, 4).sourceRanks, [0, 1, 2, 3]);
    assert.deepEqual(distributedHardwareDataFlow(gather, 4).destinationRanks, [0, 1, 2, 3]);
    assert.equal(reduceScatter.activeStream, "comm");
    assert.match(reduceScatter.hbmObject, /dW shard/);
    assert.ok(distributedHardwareRankViews(reduceScatter, 4, 2).every((rank) => rank.role === "reduce"));
  });

  it("distinguishes FSDP compute from communication phases", () => {
    const compute = fsdpHardwareSnapshot(
      fsdpPhases.find((phase) => phase.stage === "compute")!,
      "linear.weight",
      "fp32",
    );
    assert.equal(compute.activeStream, "compute");
    assert.equal(compute.link, "无链路传输");
  });

  it("maps comparison stages to the hardware path of each strategy", () => {
    assert.equal(comparisonHardwareSnapshot("ddp", "gradient-sync").activeStream, "comm");
    assert.match(comparisonHardwareSnapshot("zero-1", "optimizer-step").kernel, /broadcast/);
    assert.equal(comparisonHardwareSnapshot("zero-1", "optimizer-step").activeStream, "compute-then-comm");
    assert.match(comparisonHardwareSnapshot("fsdp", "next-forward").kernel, /All-Gather/);
  });

  it("shows the owner as the only rank computing one ZeRO parameter", () => {
    const update = shardedOptimizerHardwareSnapshot(
      shardedOptimizerPhases.find((phase) => phase.kind === "owner-update")!,
      "weight",
      2,
    );
    const ranks = distributedHardwareRankViews(update, 4, 2);
    assert.equal(ranks[2].role, "owner");
    assert.equal(ranks[2].stream, "compute");
    assert.match(ranks[2].memoryLabel, /m \+ v/);
    assert.doesNotMatch(ranks[0].memoryLabel, /m \+ v/);
    assert.equal(ranks.filter((rank) => rank.role === "owner").length, 1);
    assert.equal(ranks[2].selected, true);
    assert.deepEqual(ranks[2].memorySlots.map((slot) => slot.label), ["W", "grad", "m", "v"]);
    assert.deepEqual(ranks[0].memorySlots.map((slot) => slot.label), ["W replica", "grad"]);
    assert.equal(update.activeStream, "compute");
    assert.deepEqual(distributedHardwareDataFlow(update, 4).sourceRanks, [2]);
  });

  it("turns every ZeRO and FSDP phase into a concrete HBM-to-HBM flow", () => {
    for (const phase of shardedOptimizerPhases) {
      const snapshot = shardedOptimizerHardwareSnapshot(phase, "linear.weight", 1);
      const flow = distributedHardwareDataFlow(snapshot, 4);
      assert.ok(flow.sourceRanks.length > 0);
      assert.ok(flow.sourceObject.length > 0);
      assert.ok(flow.operator.length > 0);
      assert.ok(flow.destinationObject.length > 0);
      assert.ok(distributedHardwareRankViews(snapshot, 4, 0).every((rank) => rank.memorySlots.length > 0));
    }

    for (const phase of fsdpPhases) {
      const snapshot = fsdpHardwareSnapshot(phase, "linear.weight", "fp16");
      const flow = distributedHardwareDataFlow(snapshot, 4);
      assert.deepEqual(flow.sourceRanks, [0, 1, 2, 3]);
      assert.deepEqual(flow.destinationRanks, [0, 1, 2, 3]);
      assert.ok(flow.sourceObject.length > 0);
      assert.ok(flow.destinationObject.length > 0);
      assert.ok(distributedHardwareRankViews(snapshot, 4, 0).every((rank) => rank.memorySlots.length > 0));
    }
  });

  it("rejects invalid world sizes in the global hardware stage", () => {
    const snapshot = comparisonHardwareSnapshot("ddp", "persistent");
    assert.throws(() => distributedHardwareRankViews(snapshot, 0, 0), /worldSize/);
    assert.throws(() => distributedHardwareDataFlow(snapshot, 0), /worldSize/);
  });
});
