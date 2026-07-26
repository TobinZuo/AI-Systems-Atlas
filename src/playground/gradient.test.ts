import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  defaultGradientConfig,
  simulateGradient,
  simulateGradientDDP,
  type GradientBatchSize,
  type GradientDevice,
  type GradientDirection,
  type GradientReduction,
} from "./gradient";

const close = (actual: number, expected: number, tolerance = 1e-9) => {
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
};

describe("Gradient teaching model", () => {
  it("keeps every sample contribution visible before batch reduction", () => {
    const simulation = simulateGradient(defaultGradientConfig);
    assert.equal(simulation.samples.length, 4);
    assert.deepEqual(simulation.samples[0].gradient, { w0: -0.5, w1: -1, b: -0.5 });
    assert.deepEqual(simulation.samples[3].gradient, { w0: 4.5, w1: 1.5, b: 1.5 });
    assert.deepEqual(simulation.aggregateGradient, { w0: 1.125, w1: 0.125, b: 0.5 });
  });

  it("changes gradient scale when reduction changes from mean to sum", () => {
    const mean = simulateGradient(defaultGradientConfig);
    const sum = simulateGradient({ ...defaultGradientConfig, reduction: "sum" });
    close(sum.aggregateLoss, mean.aggregateLoss * 4);
    close(sum.aggregateGradient.w0, mean.aggregateGradient.w0 * 4);
    close(sum.aggregateGradient.w1, mean.aggregateGradient.w1 * 4);
    close(sum.aggregateGradient.b, mean.aggregateGradient.b * 4);
  });

  it("matches analytic gradients with central finite differences", () => {
    const simulation = simulateGradient(defaultGradientConfig);
    for (const check of simulation.finiteDifferences) {
      assert.ok(check.absoluteError < 1e-8, `${check.parameter} error ${check.absoluteError}`);
    }
  });

  it("moves down the local objective only when the update uses the negative gradient", () => {
    const descent = simulateGradient(defaultGradientConfig);
    const ascent = simulateGradient({ ...defaultGradientConfig, direction: "ascent" });
    assert.ok(descent.step.lossChange < 0);
    assert.ok(descent.step.predictedFirstOrderChange < 0);
    assert.ok(ascent.step.lossChange > 0);
    assert.ok(ascent.step.predictedFirstOrderChange > 0);
  });

  it("shows that a large learning rate can overshoot even along the negative gradient", () => {
    const overshoot = simulateGradient({ ...defaultGradientConfig, learningRate: 0.5 });
    assert.equal(overshoot.config.direction, "descent");
    assert.ok(overshoot.step.predictedFirstOrderChange < 0);
    assert.ok(overshoot.step.lossChange > 0);
  });

  it("stores gradient buffers beside parameters on the selected device", () => {
    const cuda = simulateGradient(defaultGradientConfig);
    const cpu = simulateGradient({ ...defaultGradientConfig, device: "cpu" });
    assert.deepEqual(cuda.buffers.map((buffer) => buffer.shape), ["[2]", "[1]"]);
    assert.deepEqual(cuda.flattenedBucket, [1.125, 0.125, 0.5]);
    assert.ok(cuda.buffers.every((buffer) => buffer.memory === "GPU HBM"));
    assert.ok(cpu.buffers.every((buffer) => buffer.memory === "CPU DRAM"));
  });

  it("matches a global sample mean only when rank-local means have equal batch weight", () => {
    const equal = simulateGradientDDP(defaultGradientConfig, "2+2");
    const uneven = simulateGradientDDP(defaultGradientConfig, "1+3");
    assert.equal(equal.rankMeanMatchesGlobal, true);
    assert.deepEqual(equal.rankMean, equal.globalBatchMean);
    assert.equal(uneven.rankMeanMatchesGlobal, false);
    assert.deepEqual(uneven.sampleWeightedMean, uneven.globalBatchMean);
  });

  it("rejects impossible teaching configurations", () => {
    assert.throws(() => simulateGradient({ ...defaultGradientConfig, learningRate: 0 }), /positive/);
    assert.throws(() => simulateGradient({ ...defaultGradientConfig, batchSize: 3 as 4 }), /batchSize/);
    assert.throws(() => simulateGradient({ ...defaultGradientConfig, reduction: "median" as "mean" }), /reduction/);
  });

  it("keeps every UI-selectable configuration finite", () => {
    let configurationCount = 0;
    for (const w0 of [-1, 0, 0.5, 1, 2]) {
      for (const w1 of [-2, -1, 0, 1]) {
        for (const b of [0, 0.5, 1]) {
          for (const batchSize of [1, 2, 4] as GradientBatchSize[]) {
            for (const reduction of ["mean", "sum"] as GradientReduction[]) {
              for (const learningRate of [0.05, 0.1, 0.5, 1]) {
                for (const direction of ["descent", "ascent"] as GradientDirection[]) {
                  for (const device of ["cpu", "cuda"] as GradientDevice[]) {
                    const simulation = simulateGradient({ w0, w1, b, batchSize, reduction, learningRate, direction, device });
                    assert.ok([
                      simulation.aggregateLoss,
                      simulation.gradientNorm,
                      simulation.step.nextLoss,
                      ...simulation.flattenedBucket,
                    ].every(Number.isFinite));
                    configurationCount += 1;
                  }
                }
              }
            }
          }
        }
      }
    }
    assert.equal(configurationCount, 5_760);
  });
});
