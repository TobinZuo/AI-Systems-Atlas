import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  adamWFirstStep,
  ddpOptimizerMemoryBytes,
  optimizerMemoryByRank,
  ownerForParameter,
  parameterValueAtPhase,
  shardedOptimizerParameters,
  shardedOptimizerPhases,
} from "./shardedOptimizer";

describe("ZeRO-1 sharded optimizer teaching model", () => {
  it("assigns each unique parameter with the round-robin owner rule", () => {
    assert.deepEqual(
      shardedOptimizerParameters.map((parameter) => ownerForParameter(parameter.index, 2)),
      [0, 1, 0, 1, 0, 1],
    );
    assert.equal(shardedOptimizerParameters.filter((parameter) => parameter.id === "tied-weight").length, 1);
  });

  it("temporarily diverges after local step and converges after broadcasts", () => {
    const parameter = shardedOptimizerParameters[0];
    const localStep = shardedOptimizerPhases.find((phase) => phase.id === "owner-update")!;
    const complete = shardedOptimizerPhases.at(-1)!;

    assert.notEqual(
      parameterValueAtPhase(parameter, 0, 2, localStep),
      parameterValueAtPhase(parameter, 1, 2, localStep),
    );
    assert.equal(
      parameterValueAtPhase(parameter, 0, 2, complete),
      parameterValueAtPhase(parameter, 1, 2, complete),
    );
  });

  it("keeps parameter-group hyperparameters in the AdamW update", () => {
    const fast = adamWFirstStep(shardedOptimizerParameters[0]);
    const decay = adamWFirstStep(shardedOptimizerParameters[3]);
    assert.ok(fast.nextValue < shardedOptimizerParameters[0].value);
    assert.ok(decay.nextValue < shardedOptimizerParameters[3].value);
    assert.notEqual(
      shardedOptimizerParameters[0].value - fast.nextValue,
      shardedOptimizerParameters[3].value - decay.nextValue,
    );
  });

  it("replicates parameters and gradients but shards the two AdamW states", () => {
    const memories = optimizerMemoryByRank(2);
    assert.equal(memories[0].parameterBytes, memories[1].parameterBytes);
    assert.equal(memories[0].gradientBytes, memories[1].gradientBytes);
    assert.equal(
      memories.reduce((sum, rank) => sum + rank.optimizerStateBytes, 0),
      ddpOptimizerMemoryBytes() / 2,
    );
    assert.ok(memories.every((rank) => rank.totalBytes < ddpOptimizerMemoryBytes()));
  });
});
