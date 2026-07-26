import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { fsdpPhases } from "./fsdp";
import { shardedOptimizerPhases } from "./shardedOptimizer";
import {
  comparisonHardwareSnapshot,
  fsdpHardwareSnapshot,
  shardedOptimizerHardwareSnapshot,
} from "./distributedHardware";

describe("distributed hardware-path teaching model", () => {
  it("runs ZeRO owner updates on compute and broadcasts on comm", () => {
    const update = shardedOptimizerHardwareSnapshot(
      shardedOptimizerPhases.find((phase) => phase.kind === "owner-update")!,
      "weight",
      0,
      0,
    );
    const broadcast = shardedOptimizerHardwareSnapshot(
      shardedOptimizerPhases.find((phase) => phase.kind === "broadcast")!,
      "weight",
      0,
      1,
    );
    assert.equal(update.activeStream, "compute");
    assert.equal(update.link, "无链路传输");
    assert.equal(broadcast.activeStream, "comm");
    assert.equal(broadcast.link, "NVLink / PCIe P2P");
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
    assert.equal(reduceScatter.activeStream, "comm");
    assert.match(reduceScatter.hbmObject, /dW shard/);
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
});
