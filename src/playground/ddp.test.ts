import assert from "node:assert/strict";
import test from "node:test";
import { simulateRingAllReduce } from "../sim/ring";
import {
  gradientPresets,
  playgroundSteps,
  ranksForStep,
  roundForStep,
  stepIndexForConceptEvent,
} from "./ddp";

test("playground maps each communication step to one ring round", () => {
  const simulation = simulateRingAllReduce(gradientPresets[0].values);
  const communicationSteps = playgroundSteps.filter(
    (step) => step.stage === "reduce-scatter" || step.stage === "all-gather",
  );
  assert.equal(communicationSteps.length, 6);
  for (const step of communicationSteps) {
    assert.ok(roundForStep(simulation, step));
  }
});

test("average step shows the same averaged tensor on every rank", () => {
  const simulation = simulateRingAllReduce(gradientPresets[0].values);
  const averageStep = playgroundSteps.find((step) => step.stage === "average");
  assert.ok(averageStep);
  const ranks = ranksForStep(simulation, averageStep);
  for (const rank of ranks) {
    assert.deepEqual(
      rank.chunks.flatMap((chunk) => chunk.values),
      simulation.averaged,
    );
  }
});

test("trace concept ids land on the matching playground phase", () => {
  assert.equal(stepIndexForConceptEvent("reduce-scatter-2"), 3);
  assert.equal(stepIndexForConceptEvent("all-gather-1"), 5);
  assert.equal(stepIndexForConceptEvent("optimizer-update"), 8);
});
