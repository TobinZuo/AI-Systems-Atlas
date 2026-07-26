import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { getAutogradNode, simulateAutograd } from "./autograd";

const baseConfig = {
  device: "cuda",
  w: 2,
  x: 3,
  b: 1,
  target: 10,
  includeRegularizer: true,
  mode: "tracked",
  backwardPasses: 1,
  zeroBetweenPasses: true,
} as const;

describe("Autograd teaching model", () => {
  it("computes forward values and sums gradient contributions from both w branches", () => {
    const simulation = simulateAutograd(baseConfig);
    const w = getAutogradNode(simulation, "w");

    assert.deepEqual(simulation.forward, {
      wx: 6,
      regularizer: 2,
      prediction: 9,
      residual: -1,
      loss: 0.5,
    });
    assert.deepEqual(simulation.onePassGradients, { w: -5, b: -1 });
    assert.deepEqual(simulation.finalGradients, { w: -5, b: -1 });
    assert.equal(w?.contributions.length, 2);
    assert.equal(w?.contributions.reduce((sum, item) => sum + item.value, 0), -5);
    assert.equal(simulation.backwardSucceeded, true);
  });

  it("changes the dynamic graph and derivative when the regularizer branch is absent", () => {
    const simulation = simulateAutograd({ ...baseConfig, includeRegularizer: false });

    assert.equal(getAutogradNode(simulation, "regularizer"), undefined);
    assert.equal(simulation.edges.some((edge) => edge.to === "regularizer"), false);
    assert.deepEqual(simulation.forward, {
      wx: 6,
      regularizer: 0,
      prediction: 7,
      residual: -3,
      loss: 4.5,
    });
    assert.deepEqual(simulation.onePassGradients, { w: -9, b: -3 });
    assert.equal(getAutogradNode(simulation, "w")?.contributions.length, 1);
  });

  it("keeps the differentiation result device independent", () => {
    const cuda = simulateAutograd(baseConfig);
    const cpu = simulateAutograd({ ...baseConfig, device: "cpu" });

    assert.deepEqual(cpu.forward, cuda.forward);
    assert.deepEqual(cpu.finalGradients, cuda.finalGradients);
    assert.match(getAutogradNode(cuda, "w")!.expression, /device="cuda"/);
    assert.match(getAutogradNode(cpu, "w")!.expression, /device="cpu"/);
  });

  it("shows leaf gradients accumulating across separately rebuilt graphs", () => {
    const accumulated = simulateAutograd({
      ...baseConfig,
      backwardPasses: 2,
      zeroBetweenPasses: false,
    });
    const cleared = simulateAutograd({
      ...baseConfig,
      backwardPasses: 2,
      zeroBetweenPasses: true,
    });

    assert.equal(accumulated.graphGenerationCount, 2);
    assert.equal(accumulated.gradientMultiplier, 2);
    assert.deepEqual(accumulated.finalGradients, { w: -10, b: -2 });
    assert.equal(cleared.gradientMultiplier, 1);
    assert.deepEqual(cleared.finalGradients, { w: -5, b: -1 });
  });

  it("computes forward values but records no graph in no-grad mode", () => {
    const simulation = simulateAutograd({ ...baseConfig, mode: "no-grad" });

    assert.equal(simulation.forward.loss, 0.5);
    assert.equal(simulation.graphRecorded, false);
    assert.equal(getAutogradNode(simulation, "loss")?.gradFn, null);
    assert.equal(getAutogradNode(simulation, "loss")?.requiresGrad, false);
    assert.equal(simulation.savedTensorCount, 0);
    assert.equal(simulation.backwardSucceeded, false);
    assert.equal(simulation.errorCode, "requires_grad=false");
    assert.equal(simulation.finalGradients, null);
  });

  it("detects a saved tensor version mismatch after an in-place mutation", () => {
    const simulation = simulateAutograd({ ...baseConfig, mode: "in-place" });

    assert.equal(simulation.graphRecorded, true);
    assert.equal(getAutogradNode(simulation, "w")?.version, 1);
    assert.ok(getAutogradNode(simulation, "regularizer")?.saved.includes("saved version = 0"));
    assert.equal(simulation.backwardSucceeded, false);
    assert.equal(simulation.errorCode, "saved tensor version mismatch");
    assert.equal(simulation.steps.find((step) => step.id === "version-check-fault")?.status, "fault");
    assert.equal(simulation.finalGradients, null);
  });

  it("rejects invalid values and impossible teaching configurations", () => {
    assert.throws(() => simulateAutograd({ ...baseConfig, w: Number.NaN }), /w must be finite/);
    assert.throws(
      () => simulateAutograd({ ...baseConfig, mode: "in-place", includeRegularizer: false }),
      /regularizer branch/,
    );
    assert.throws(
      () => simulateAutograd({ ...baseConfig, mode: "no-grad", backwardPasses: 2 }),
      /one backward request/,
    );
    assert.throws(
      () => simulateAutograd({ ...baseConfig, device: "tpu" as "cpu" }),
      /device must be cpu or cuda/,
    );
  });
});
