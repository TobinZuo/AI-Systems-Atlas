import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  ATTENTION_TOKENS,
  attentionCost,
  attentionHeadLayout,
  attentionMatrixForPhase,
  createAttentionSimulation,
  multiplyMatrices,
  selectedAttentionDetail,
  transposeMatrix,
} from "./attention";

function approximatelyEqual(actual: number, expected: number, tolerance = 1e-9): void {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected}`);
}

describe("Attention teaching model", () => {
  it("projects one input sequence into deterministic Q, K, and V roles", () => {
    const simulation = createAttentionSimulation();

    assert.deepEqual(simulation.tokens, [...ATTENTION_TOKENS]);
    assert.deepEqual(simulation.q, [
      [1, 0, 1],
      [0, 1, 1],
      [1, 1, 0],
      [1, 1, 1],
    ]);
    assert.deepEqual(simulation.k, [
      [1, 1, 0],
      [0, 1, 1],
      [1, 0, 1],
      [1, 1, 1],
    ]);
    assert.deepEqual(simulation.v, [[2, 1], [1, 2], [1, 1], [2, 2]]);
  });

  it("computes every pairwise Q dot K score", () => {
    const simulation = createAttentionSimulation("bidirectional", "unscaled");

    assert.deepEqual(simulation.rawScores, [
      [1, 1, 2, 2],
      [1, 2, 1, 2],
      [2, 1, 1, 2],
      [2, 2, 2, 3],
    ]);
  });

  it("applies a strict upper-triangular causal mask before row-wise softmax", () => {
    const simulation = createAttentionSimulation("causal", "scaled");

    assert.deepEqual(simulation.maskedScores.map((row) => row.map((value) => value === null)), [
      [false, true, true, true],
      [false, false, true, true],
      [false, false, false, true],
      [false, false, false, false],
    ]);
    simulation.weights.forEach((row, queryIndex) => {
      approximatelyEqual(row.reduce((sum, value) => sum + value, 0), 1);
      row.forEach((weight, keyIndex) => {
        if (keyIndex > queryIndex) assert.equal(weight, 0);
      });
    });
  });

  it("scaling makes the same score row less concentrated before training", () => {
    const scaled = createAttentionSimulation("bidirectional", "scaled");
    const unscaled = createAttentionSimulation("bidirectional", "unscaled");

    assert.ok(Math.max(...scaled.weights[3]) < Math.max(...unscaled.weights[3]));
    approximatelyEqual(scaled.scaledScores[3][3], 3 / Math.sqrt(3));
  });

  it("mixes Value vectors with the selected query row of probabilities", () => {
    const simulation = createAttentionSimulation("causal", "scaled");
    const detail = selectedAttentionDetail(simulation, 2, 0);
    const expected = simulation.weights[2].reduce(
      (sum, weight, keyIndex) => [
        sum[0] + weight * simulation.v[keyIndex][0],
        sum[1] + weight * simulation.v[keyIndex][1],
      ],
      [0, 0],
    );

    approximatelyEqual(detail.outputVector[0], expected[0]);
    approximatelyEqual(detail.outputVector[1], expected[1]);
    assert.deepEqual(detail.valueVector, [2, 1]);
    assert.equal(selectedAttentionDetail(simulation, 1, 3).masked, true);
  });

  it("exposes the correct matrix at each teaching phase", () => {
    const simulation = createAttentionSimulation();

    assert.equal(attentionMatrixForPhase(simulation, "input"), null);
    assert.deepEqual(attentionMatrixForPhase(simulation, "score"), simulation.rawScores);
    assert.deepEqual(attentionMatrixForPhase(simulation, "mask"), simulation.maskedScores);
    assert.deepEqual(attentionMatrixForPhase(simulation, "mix"), simulation.weights);
  });

  it("reshapes a projection into independent heads and concatenates back", () => {
    const layout = attentionHeadLayout(8, 4, 16);

    assert.deepEqual(layout.projectedShape, [16, 8]);
    assert.deepEqual(layout.headShape, [4, 16, 2]);
    assert.equal(layout.scoreMatrices, 4);
    assert.equal(layout.scoreElements, 4 * 16 * 16);
    assert.deepEqual(layout.concatenatedShape, [16, 8]);
  });

  it("shows quadratic score growth while QKV storage grows linearly", () => {
    const short = attentionCost(128, 8, 64);
    const long = attentionCost(256, 8, 64);

    assert.equal(long.scoreElements, short.scoreElements * 4);
    assert.equal(long.qkvElements, short.qkvElements * 2);
    assert.equal(short.totalAttentionFlops, short.qkFlops + short.avFlops);
    assert.equal(short.causalUsefulScoreElements, 8 * 128 * 129 / 2);
  });

  it("validates matrix shapes, indices, head layouts, and costs", () => {
    assert.deepEqual(multiplyMatrices([[1, 2]], [[3], [4]]), [[11]]);
    assert.deepEqual(transposeMatrix([[1, 2], [3, 4]]), [[1, 3], [2, 4]]);
    assert.throws(() => multiplyMatrices([[1, 2]], [[1, 2]]), /inner dimensions/);
    assert.throws(() => selectedAttentionDetail(createAttentionSimulation(), 4, 0), /queryIndex/);
    assert.throws(() => attentionHeadLayout(10, 4, 8), /divisible/);
    assert.throws(() => attentionCost(0, 8, 64), /sequenceLength/);
  });
});
