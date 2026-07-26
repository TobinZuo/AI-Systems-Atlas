import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  MEMORY_DW,
  effectiveMemoryTileSize,
  gradientEquation,
  gradientTile,
  memoryResidency,
  memorySteps,
  memoryTraffic,
  multiplyWeightGradient,
} from "./memoryHierarchy";

describe("GPU memory hierarchy teaching model", () => {
  it("computes the exact linear-weight gradient dW = X transpose times dY", () => {
    assert.deepEqual(MEMORY_DW, [
      [0, 4, 5, -1],
      [4, 2, 5, 2],
      [8, 0, 5, 5],
      [4, 4, 9, 6],
    ]);
    assert.deepEqual(multiplyWeightGradient(), MEMORY_DW);
  });

  it("keeps every selected output tied to four visible products", () => {
    const equation = gradientEquation(2, 3);

    assert.deepEqual(equation.terms.map((term) => term.product), [3, 0, 2, 0]);
    assert.equal(equation.result, 5);
  });

  it("builds the two input tiles and output tile around a selected dW element", () => {
    const tile = gradientTile(3, 2, 0, 2);

    assert.deepEqual(tile.outputRows, [2, 3]);
    assert.deepEqual(tile.outputColumns, [2, 3]);
    assert.deepEqual(tile.kValues, [0, 1]);
    assert.deepEqual(tile.xValues.map((value) => value.id), ["X[0,2]", "X[0,3]", "X[1,2]", "X[1,3]"]);
    assert.deepEqual(tile.dyValues.map((value) => value.id), ["dY[0,2]", "dY[0,3]", "dY[1,2]", "dY[1,3]"]);
    assert.equal(tile.outputValues.filter((value) => value.selectedOperand).length, 1);
  });

  it("tracks the accumulator across aligned K tiles", () => {
    const first = gradientTile(3, 2, 0, 2);
    const second = gradientTile(3, 2, 2, 2);

    assert.equal(first.accumulatorBefore, 0);
    assert.equal(first.currentContribution, 7);
    assert.equal(first.accumulatorAfter, 7);
    assert.equal(second.accumulatorBefore, 7);
    assert.equal(second.currentContribution, 2);
    assert.equal(second.accumulatorAfter, MEMORY_DW[3][2]);
  });

  it("removes the explicit shared-memory stage from the naive path", () => {
    assert.deepEqual(memorySteps("naive").map((step) => step.id), [
      "global-request",
      "l2-lookup",
      "register-load",
      "fma",
      "writeback",
    ]);
    assert.equal(memorySteps("tiled").some((step) => step.id === "shared-stage"), true);
    assert.equal(effectiveMemoryTileSize("naive", 4), 1);
  });

  it("moves one tile through cache, shared memory, registers, and writeback", () => {
    const atL2 = memoryResidency("tiled", "l2-lookup");
    const atFma = memoryResidency("tiled", "fma");
    const complete = memoryResidency("tiled", "writeback");

    assert.equal(atL2.l2Tile, true);
    assert.equal(atL2.sharedTile, false);
    assert.equal(atFma.sharedTile, true);
    assert.equal(atFma.registerOperands, true);
    assert.equal(atFma.accumulator, true);
    assert.equal(atFma.hbmOutput, false);
    assert.equal(complete.hbmOutput, true);
  });

  it("shows how larger tiles trade on-chip reuse for fewer global loads", () => {
    const naive = memoryTraffic("naive", 2);
    const tile2 = memoryTraffic("tiled", 2);
    const tile4 = memoryTraffic("tiled", 4);

    assert.equal(naive.globalReadScalars, 128);
    assert.equal(tile2.globalReadScalars, 64);
    assert.equal(tile4.globalReadScalars, 32);
    assert.equal(naive.globalWriteScalars, 16);
    assert.equal(tile2.reusePerStagedScalar, 2);
    assert.equal(tile4.reusePerStagedScalar, 4);
    assert.ok(naive.arithmeticIntensity < tile2.arithmeticIntensity);
    assert.ok(tile2.arithmeticIntensity < tile4.arithmeticIntensity);
  });

  it("rejects invalid coordinates, tiles, and matrices", () => {
    assert.throws(() => gradientEquation(-1, 0), /row/);
    assert.throws(() => gradientTile(0, 0, 1, 2), /kStart/);
    assert.throws(() => memoryTraffic("tiled", 3), /tileSize/);
    assert.throws(() => multiplyWeightGradient([[1, 2]], [[1, 2]]), /square/);
  });
});
