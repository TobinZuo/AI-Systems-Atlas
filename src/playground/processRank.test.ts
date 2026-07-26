import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  collectiveDataPath,
  defaultProcessRankConfig,
  processInitPhases,
  processStatesAtPhase,
  simulateProcessRanks,
  type ClusterLayout,
  type ProcessBackend,
  type ProcessInitFault,
} from "./processRank";

describe("Process, rank, and process-group teaching model", () => {
  it("maps four global ranks to unique local GPUs on two nodes", () => {
    const simulation = simulateProcessRanks(defaultProcessRankConfig);
    assert.deepEqual(simulation.workers.map((worker) => [worker.hostname, worker.globalRank, worker.localRank, worker.device]), [
      ["node-0", 0, 0, "cuda:0"],
      ["node-0", 1, 1, "cuda:1"],
      ["node-1", 2, 0, "cuda:0"],
      ["node-1", 3, 1, "cuda:1"],
    ]);
  });

  it("creates one local default-group object per process, not one cross-process singleton", () => {
    const simulation = simulateProcessRanks(defaultProcessRankConfig);
    assert.equal(new Set(simulation.workers.map((worker) => worker.defaultGroupObject)).size, 4);
    assert.equal(simulation.logicalGroupId, "default_group / world");
    assert.ok(simulation.workers.every((worker) => worker.environment.WORLD_SIZE === "4"));
  });

  it("uses the master endpoint for rendezvous but not for collective payload", () => {
    const simulation = simulateProcessRanks(defaultProcessRankConfig);
    const localPath = collectiveDataPath(simulation, 0);
    const crossNodePath = collectiveDataPath(simulation, 1);
    assert.equal(localPath.transport, "NVLink / PCIe P2P");
    assert.equal(crossNodePath.transport, "InfiniBand / RoCE / IP network");
    assert.equal(localPath.usesMasterPortForPayload, false);
    assert.equal(crossNodePath.usesMasterPortForPayload, false);
  });

  it("routes Gloo CPU tensors through TCP sockets", () => {
    const simulation = simulateProcessRanks({ ...defaultProcessRankConfig, backend: "gloo" });
    const path = collectiveDataPath(simulation, 1);
    assert.equal(path.sourceMemory, "CPU DRAM");
    assert.equal(path.collectiveLayer, "ProcessGroupGloo");
    assert.equal(path.transport, "TCP sockets");
  });

  it("stops every rank after a mismatched rendezvous contract", () => {
    for (const fault of ["duplicate-rank", "world-size-mismatch", "master-port-mismatch"] as ProcessInitFault[]) {
      const simulation = simulateProcessRanks({ ...defaultProcessRankConfig, fault });
      const rendezvous = processStatesAtPhase(simulation, "rendezvous");
      const ready = processStatesAtPhase(simulation, "ready");
      assert.ok(rendezvous.some((worker) => worker.state === "error"));
      assert.ok(ready.every((worker) => worker.state === "blocked"));
    }
  });

  it("treats duplicate CUDA devices as an NCCL communicator failure", () => {
    const nccl = simulateProcessRanks({ ...defaultProcessRankConfig, fault: "device-collision" });
    const gloo = simulateProcessRanks({ ...defaultProcessRankConfig, backend: "gloo", fault: "device-collision" });
    assert.equal(nccl.fault.failurePhase, "communicator");
    assert.equal(gloo.fault.failurePhase, null);
    assert.ok(processStatesAtPhase(nccl, "ready").every((worker) => worker.state === "blocked"));
    assert.ok(processStatesAtPhase(gloo, "ready").every((worker) => worker.state === "ready"));
  });

  it("keeps every selectable topology and phase deterministic", () => {
    const layouts: ClusterLayout[] = ["single-node", "two-node"];
    const backends: ProcessBackend[] = ["nccl", "gloo"];
    const faults: ProcessInitFault[] = ["none", "duplicate-rank", "world-size-mismatch", "master-port-mismatch", "device-collision"];
    for (const layout of layouts) for (const backend of backends) for (const fault of faults) {
      const simulation = simulateProcessRanks({ layout, backend, fault });
      assert.equal(simulation.workers.length, 4);
      for (const phase of processInitPhases) assert.equal(processStatesAtPhase(simulation, phase.id).length, 4);
      for (let rank = 0; rank < 4; rank += 1) assert.equal(collectiveDataPath(simulation, rank).sourceRank, rank);
    }
  });

  it("rejects ranks outside the process group", () => {
    const simulation = simulateProcessRanks(defaultProcessRankConfig);
    assert.throws(() => collectiveDataPath(simulation, -1));
    assert.throws(() => collectiveDataPath(simulation, 4));
  });
});
