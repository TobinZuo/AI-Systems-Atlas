import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  comparisonPhaseInsights,
  distributedCommunicationSnapshots,
  distributedMemorySnapshots,
  recommendDistributedStrategy,
} from "./distributedComparison";

describe("distributed strategy comparison model", () => {
  it("keeps all FP32 model states replicated under DDP", () => {
    const [ddp] = distributedMemorySnapshots(1_000, 4);
    assert.equal(ddp.parameterBytes, 4_000);
    assert.equal(ddp.gradientBytes, 4_000);
    assert.equal(ddp.optimizerStateBytes, 8_000);
    assert.equal(ddp.totalBytes, 16_000);
  });

  it("only shards optimizer state under ZeRO-1", () => {
    const snapshots = distributedMemorySnapshots(1_000, 4);
    const zero = snapshots.find((snapshot) => snapshot.id === "zero-1")!;
    assert.equal(zero.parameterFraction, 1);
    assert.equal(zero.gradientFraction, 1);
    assert.equal(zero.optimizerStateFraction, 0.25);
    assert.equal(zero.totalBytes, 10_000);
  });

  it("shards parameters, gradients, and optimizer state under FSDP", () => {
    const snapshots = distributedMemorySnapshots(1_000, 4);
    const fsdp = snapshots.find((snapshot) => snapshot.id === "fsdp")!;
    assert.equal(fsdp.parameterFraction, 0.25);
    assert.equal(fsdp.gradientFraction, 0.25);
    assert.equal(fsdp.optimizerStateFraction, 0.25);
    assert.equal(fsdp.totalBytes, 4_000);
  });

  it("reduces every strategy to the same state size for one rank", () => {
    const totals = distributedMemorySnapshots(1_000, 1).map((snapshot) => snapshot.totalBytes);
    assert.deepEqual(totals, [16_000, 16_000, 16_000]);
  });

  it("models ring payloads and owner parameter receipts separately", () => {
    const profiles = distributedCommunicationSnapshots(1_000, 4);
    const ddp = profiles.find((profile) => profile.id === "ddp")!;
    const zero = profiles.find((profile) => profile.id === "zero-1")!;
    const fsdp = profiles.find((profile) => profile.id === "fsdp")!;
    assert.equal(ddp.gradientReceiveBytes, 6_000);
    assert.equal(zero.parameterReceiveBytes, 3_000);
    assert.equal(fsdp.parameterReceiveBytes, 6_000);
    assert.equal(fsdp.gradientReceiveBytes, 3_000);
    assert.equal(zero.totalReceiveBytes, fsdp.totalReceiveBytes);
  });

  it("explains the distinct optimizer-step contracts", () => {
    const insights = comparisonPhaseInsights("optimizer-step");
    assert.match(insights.find((item) => item.id === "ddp")!.communication, /无需同步/);
    assert.match(insights.find((item) => item.id === "zero-1")!.communication, /Broadcast/);
    assert.match(insights.find((item) => item.id === "fsdp")!.dataState, /1\/N/);
  });

  it("maps memory pressure to a strategy without claiming universal optimality", () => {
    assert.equal(recommendDistributedStrategy("fits").strategy, "ddp");
    assert.equal(recommendDistributedStrategy("optimizer-pressure").strategy, "zero-1");
    assert.equal(recommendDistributedStrategy("model-pressure").strategy, "fsdp");
  });
});
