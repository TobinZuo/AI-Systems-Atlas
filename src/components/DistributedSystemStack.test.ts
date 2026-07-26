import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { distributedSystemStackProfiles } from "./DistributedSystemStack";

describe("Distributed system stack profiles", () => {
  it("shows the same six system layers for every distributed topic", () => {
    const expectedLayers = ["framework", "process", "stream", "kernel", "memory", "fabric"];

    for (const [topic, profile] of Object.entries(distributedSystemStackProfiles)) {
      assert.deepEqual(Object.keys(profile.layers), expectedLayers, `${topic} has a different system stack`);
    }
  });

  it("distinguishes rendezvous from active GPU communication", () => {
    const rendezvous = distributedSystemStackProfiles["process-rank"];
    assert.equal(rendezvous.layers.stream.state, "deferred");
    assert.equal(rendezvous.layers.kernel.state, "deferred");

    for (const topic of ["collective", "ddp", "zero-1", "fsdp"] as const) {
      assert.equal(distributedSystemStackProfiles[topic].layers.fabric.state, "active");
    }
  });

  it("gives every profile a concrete drill-down target", () => {
    const targets = Object.values(distributedSystemStackProfiles).map((profile) => profile.targetId);
    assert.equal(new Set(targets).size, targets.length);
    assert.ok(targets.every(Boolean));
  });
});
