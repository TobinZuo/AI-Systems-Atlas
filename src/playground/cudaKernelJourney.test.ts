import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { firstFaultIndex, simulateKernelJourney } from "./cudaKernelJourney";

const cudaConfig = {
  device: "cuda",
  vectorLength: 70,
  threadsPerBlock: 32,
  scale: 0.5,
  fault: "none",
} as const;

describe("CUDA kernel journey teaching model", () => {
  it("dispatches CUDA tensors to a launch with pointers and scalar arguments", () => {
    const simulation = simulateKernelJourney(cudaConfig);

    assert.equal(simulation.selectedBackend, "CUDA kernel");
    assert.equal(simulation.gridDim, 3);
    assert.equal(simulation.launchedThreadCount, 96);
    assert.equal(simulation.usefulThreadCount, 70);
    assert.equal(simulation.maskedThreadCount, 26);
    assert.equal(simulation.arguments.filter((argument) => argument.kind === "pointer").length, 2);
    assert.equal(simulation.resultValid, true);
    assert.equal(simulation.hostReturnsBeforeCompletion, true);
    assert.ok(simulation.gpu);
  });

  it("bypasses CUDA runtime, streams, and device execution for a CPU tensor", () => {
    const simulation = simulateKernelJourney({ ...cudaConfig, device: "cpu" });

    assert.equal(simulation.selectedBackend, "CPU kernel");
    assert.equal(simulation.gridDim, null);
    assert.equal(simulation.hostReturnsBeforeCompletion, false);
    assert.equal(simulation.gpu, null);
    for (const stepId of ["runtime-launch", "stream-enqueue", "device-execute"]) {
      assert.equal(simulation.steps.find((item) => item.id === stepId)?.status, "bypassed");
    }
  });

  it("stops before enqueue when launch configuration is invalid", () => {
    const simulation = simulateKernelJourney({ ...cudaConfig, fault: "invalid-launch" });

    assert.equal(simulation.effectiveThreadsPerBlock, 2048);
    assert.equal(simulation.launchAccepted, false);
    assert.equal(simulation.errorCode, "cudaErrorInvalidConfiguration");
    assert.equal(simulation.errorObservedAt, "immediate launch check");
    assert.equal(simulation.steps.find((item) => item.id === "runtime-launch")?.status, "fault");
    assert.equal(simulation.steps.find((item) => item.id === "device-execute")?.status, "blocked");
    assert.equal(simulation.resultValid, false);
  });

  it("surfaces a device memory fault at a later synchronization boundary", () => {
    const simulation = simulateKernelJourney({ ...cudaConfig, fault: "device-memory" });

    assert.equal(simulation.launchAccepted, true);
    assert.equal(simulation.steps.find((item) => item.id === "runtime-launch")?.status, "executed");
    assert.equal(simulation.steps.find((item) => item.id === "device-execute")?.status, "fault");
    assert.equal(simulation.errorCode, "cudaErrorIllegalAddress");
    assert.equal(simulation.errorObservedAt, "later synchronization");
    assert.equal(simulation.resultValid, false);
    assert.equal(firstFaultIndex(simulation), 6);
  });

  it("rejects impossible configs and CUDA-only faults on CPU", () => {
    assert.throws(() => simulateKernelJourney({ ...cudaConfig, vectorLength: 0 }), /vectorLength/);
    assert.throws(() => simulateKernelJourney({ ...cudaConfig, threadsPerBlock: 2048 }), /threadsPerBlock/);
    assert.throws(() => simulateKernelJourney({ ...cudaConfig, device: "cpu", fault: "device-memory" }), /CUDA faults/);
  });
});
