import assert from "node:assert/strict";
import test from "node:test";
import { simulateRingAllReduce } from "../sim/ring";
import {
  chunkJourney,
  gradientPresets,
  playgroundSteps,
  ranksBeforeStep,
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

test("a followed chunk keeps one continuous route through both collectives", () => {
  const simulation = simulateRingAllReduce(gradientPresets[0].values);
  const journey = chunkJourney(simulation, 2);

  assert.deepEqual(
    journey.map(({ phase, from, to }) => [phase, from, to]),
    [
      ["reduce-scatter", 2, 3],
      ["reduce-scatter", 3, 0],
      ["reduce-scatter", 0, 1],
      ["all-gather", 1, 2],
      ["all-gather", 2, 3],
      ["all-gather", 3, 0],
    ],
  );
});

test("the hardware stage can show memory before the first All-Gather copy lands", () => {
  const simulation = simulateRingAllReduce(gradientPresets[0].values);
  const firstGather = playgroundSteps.find((step) => step.id === "ag-0");
  assert.ok(firstGather);

  const before = ranksBeforeStep(simulation, firstGather);
  for (const rank of before) {
    const populated = rank.chunks.filter((chunk) => chunk.values.length > 0);
    assert.equal(populated.length, 1);
    assert.equal(populated[0].chunk, (rank.rank + 1) % simulation.worldSize);
    assert.equal(populated[0].complete, true);
  }
});
