import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  adamWParameters,
  defaultAdamWConfig,
  getAdamWMemoryBreakdown,
  simulateAdamW,
  type AdamWConfig,
} from "./adamw";

describe("AdamW teaching model", () => {
  it("matches the first-step bias-corrected update", () => {
    const trace = simulateAdamW(defaultAdamWConfig).steps[0].parameters[0];
    assert.ok(Math.abs(trace.correctedAvg - trace.gradient) < 1e-12);
    assert.ok(Math.abs(trace.correctedAvgSq - trace.gradient ** 2) < 1e-12);
    assert.ok(Math.abs(trace.gradientUpdate + 0.1) < 1e-8);
    assert.ok(Math.abs(trace.decayUpdate + 0.01) < 1e-12);
    assert.ok(Math.abs(trace.parameterAfter - 0.89) < 1e-8);
  });

  it("smooths sign changes in m and keeps v nonnegative", () => {
    const simulation = simulateAdamW(defaultAdamWConfig);
    const weight0 = simulation.steps.map((step) => step.parameters[0]);
    assert.ok(weight0[2].gradient < 0);
    assert.ok(weight0[2].expAvg > 0);
    assert.ok(simulation.steps.every((step) => step.parameters.every((parameter) => parameter.expAvgSq >= 0)));
  });

  it("keeps AdamW moments independent from weight decay", () => {
    const noDecay = simulateAdamW({ ...defaultAdamWConfig, weightDecay: 0 });
    const strongDecay = simulateAdamW({ ...defaultAdamWConfig, weightDecay: 0.5 });
    for (let step = 0; step < noDecay.steps.length; step += 1) {
      for (let parameter = 0; parameter < adamWParameters.length; parameter += 1) {
        assert.equal(noDecay.steps[step].parameters[parameter].expAvg, strongDecay.steps[step].parameters[parameter].expAvg);
        assert.equal(noDecay.steps[step].parameters[parameter].expAvgSq, strongDecay.steps[step].parameters[parameter].expAvgSq);
      }
    }
  });

  it("couples the L2 penalty into Adam moments but leaves the no-decay bias unchanged", () => {
    const adamw = simulateAdamW(defaultAdamWConfig);
    const coupled = simulateAdamW({ ...defaultAdamWConfig, mode: "coupled-l2" });
    assert.notEqual(adamw.steps[0].parameters[0].expAvg, coupled.steps[0].parameters[0].expAvg);
    for (let step = 0; step < adamw.steps.length; step += 1) {
      assert.deepEqual(adamw.steps[step].parameters[2], coupled.steps[step].parameters[2]);
    }
  });

  it("produces finite traces for every selectable control combination", () => {
    const beta1s = [0, 0.9, 0.99];
    const beta2s = [0, 0.9, 0.999];
    const learningRates = [0.001, 0.01, 0.1];
    const weightDecays = [0, 0.01, 0.1];
    const epsilons = [1e-8, 1e-4];
    const modes: AdamWConfig["mode"][] = ["adamw", "coupled-l2"];

    for (const beta1 of beta1s) for (const beta2 of beta2s) {
      for (const learningRate of learningRates) for (const weightDecay of weightDecays) {
        for (const epsilon of epsilons) for (const mode of modes) {
          const simulation = simulateAdamW({ beta1, beta2, learningRate, weightDecay, epsilon, mode });
          for (const step of simulation.steps) for (const parameter of step.parameters) {
            for (const value of Object.values(parameter)) {
              if (typeof value === "number") assert.ok(Number.isFinite(value));
            }
          }
        }
      }
    }
  });

  it("shows how DDP, ZeRO-1, and FSDP change persistent bytes per rank", () => {
    const ddp = getAdamWMemoryBreakdown("ddp", 1_000_000_000, 4);
    const zero = getAdamWMemoryBreakdown("zero-1", 1_000_000_000, 4);
    const fsdp = getAdamWMemoryBreakdown("fsdp", 1_000_000_000, 4);
    assert.equal(ddp.persistentBytesPerRank, 16_000_000_000);
    assert.equal(zero.persistentBytesPerRank, 10_000_000_000);
    assert.equal(fsdp.persistentBytesPerRank, 4_000_000_000);
    assert.deepEqual(zero.shardedComponents, ["m", "v"]);
    assert.deepEqual(fsdp.shardedComponents, ["parameter", "gradient", "m", "v"]);
  });

  it("rejects invalid optimizer hyperparameters", () => {
    assert.throws(() => simulateAdamW({ ...defaultAdamWConfig, beta1: 1 }));
    assert.throws(() => simulateAdamW({ ...defaultAdamWConfig, epsilon: 0 }));
    assert.throws(() => getAdamWMemoryBreakdown("ddp", 0, 4));
  });
});
