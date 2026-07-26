import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  CUDA_WARP_SIZE,
  formatDeviceAddress,
  selectedKernelLane,
  simulateGradientScaleKernel,
} from "./gpuArchitecture";

describe("GPU architecture teaching model", () => {
  it("expands a vector into blocks, warps, and globally indexed lanes", () => {
    const simulation = simulateGradientScaleKernel({
      vectorLength: 70,
      threadsPerBlock: 32,
      smCount: 2,
      scale: 0.5,
    });

    assert.equal(simulation.gridDim, 3);
    assert.equal(simulation.warpsPerBlock, 1);
    assert.equal(simulation.totalProgramThreads, 96);
    assert.equal(simulation.totalScheduledLanes, 96);
    assert.deepEqual(simulation.blocks.map((block) => block.smId), [0, 1, 0]);
    assert.equal(simulation.blocks[2].warps[0].activeLaneCount, 6);
    assert.equal(simulation.blocks[2].warps[0].guardedLaneCount, 26);
  });

  it("uses the CUDA global index formula and masks tail threads", () => {
    const simulation = simulateGradientScaleKernel({
      vectorLength: 70,
      threadsPerBlock: 32,
      smCount: 2,
      scale: 2,
    });
    const active = selectedKernelLane(simulation, 2, 0, 5);
    const guarded = selectedKernelLane(simulation, 2, 0, 6);

    assert.equal(active.globalIndex, 2 * 32 + 5);
    assert.equal(active.state, "active");
    assert.equal(active.outputValue, active.inputValue! * 2);
    assert.equal(guarded.globalIndex, 70);
    assert.equal(guarded.state, "guarded");
    assert.equal(guarded.inputAddress, null);
  });

  it("represents partial warps when blockDim is not a multiple of 32", () => {
    const simulation = simulateGradientScaleKernel({
      vectorLength: 48,
      threadsPerBlock: 48,
      smCount: 1,
      scale: 1,
    });
    const secondWarp = simulation.blocks[0].warps[1];

    assert.equal(simulation.warpsPerBlock, 2);
    assert.equal(secondWarp.activeLaneCount, 16);
    assert.equal(secondWarp.unusedLaneCount, CUDA_WARP_SIZE - 16);
    assert.equal(secondWarp.guardedLaneCount, 0);
  });

  it("groups adjacent float addresses into 32-byte teaching segments", () => {
    const simulation = simulateGradientScaleKernel({
      vectorLength: 32,
      threadsPerBlock: 32,
      smCount: 1,
      scale: 1,
    });
    const warp = simulation.blocks[0].warps[0];

    assert.equal(warp.inputTransactionCount, 4);
    assert.equal(warp.outputTransactionCount, 4);
    assert.equal(formatDeviceAddress(warp.lanes[0].inputAddress), "0x1000");
    assert.equal(formatDeviceAddress(null), "没有地址");
  });

  it("rejects invalid launch configurations", () => {
    assert.throws(
      () => simulateGradientScaleKernel({ vectorLength: 0, threadsPerBlock: 32, smCount: 1, scale: 1 }),
      /vectorLength/,
    );
    assert.throws(
      () => simulateGradientScaleKernel({ vectorLength: 32, threadsPerBlock: 0, smCount: 1, scale: 1 }),
      /threadsPerBlock/,
    );
  });
});
