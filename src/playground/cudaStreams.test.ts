import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  activeCudaStreamEvents,
  bucketSnapshotsAt,
  simulateCudaStreams,
} from "./cudaStreams";

const baseConfig = { bucketCount: 3, communicationDuration: 2.4 } as const;

describe("CUDA Stream teaching model", () => {
  it("serializes backward and collective work on one stream", () => {
    const simulation = simulateCudaStreams({ ...baseConfig, mode: "single" });
    const collectives = simulation.events.filter((event) => event.kind === "collective");

    assert.ok(collectives.every((event) => event.laneId === "compute"));
    assert.equal(simulation.overlapDuration, 0);
    assert.equal(simulation.speedup, 1);
    assert.equal(simulation.safe, true);

    assert.equal(
      simulation.events.filter((event) => event.id.startsWith("enqueue-collective-")).length,
      baseConfig.bucketCount,
    );

    const deviceWork = simulation.events
      .filter((event) => event.laneId === "compute" && event.duration > 0)
      .sort((left, right) => left.start - right.start);
    for (let index = 1; index < deviceWork.length; index += 1) {
      assert.ok(deviceWork[index].start + 1e-9 >= deviceWork[index - 1].start + deviceWork[index - 1].duration);
    }
  });

  it("uses Events to overlap safe communication with later backward kernels", () => {
    const simulation = simulateCudaStreams({ ...baseConfig, mode: "dual-safe" });

    for (let bucketId = 0; bucketId < baseConfig.bucketCount; bucketId += 1) {
      const producer = simulation.events.find((event) => event.id === `backward-b${bucketId}`)!;
      const collective = simulation.events.find((event) => event.id === `allreduce-b${bucketId}`)!;
      assert.ok(collective.start + 1e-9 >= producer.start + producer.duration);
      assert.ok(simulation.events.some((event) => event.id === `record-e${bucketId}`));
      assert.ok(simulation.events.some((event) => event.id === `wait-e${bucketId}`));
    }
    assert.ok(simulation.overlapDuration > 0);
    assert.ok(simulation.totalDuration < simulation.serialDuration);
    assert.ok(simulation.speedup > 1);
    assert.equal(simulation.safe, true);
  });

  it("exposes a race when the communication stream does not wait for producers", () => {
    const simulation = simulateCudaStreams({ ...baseConfig, mode: "dual-unsafe" });

    assert.equal(simulation.safe, false);
    assert.ok(simulation.hazardBucketIds.length > 0);
    for (const bucketId of simulation.hazardBucketIds) {
      const producer = simulation.events.find((event) => event.id === `backward-b${bucketId}`)!;
      const collective = simulation.events.find((event) => event.id === `allreduce-b${bucketId}`)!;
      assert.ok(collective.start < producer.start + producer.duration);
      assert.equal(collective.hazard, true);
    }
  });

  it("tracks bucket state at an arbitrary timeline cursor", () => {
    const simulation = simulateCudaStreams({ ...baseConfig, mode: "dual-safe" });
    const collective = simulation.events.find((event) => event.id === "allreduce-b2")!;
    const during = bucketSnapshotsAt(simulation, collective.start + 0.2);
    const after = bucketSnapshotsAt(simulation, simulation.totalDuration);

    assert.equal(during.find((bucket) => bucket.bucketId === 2)?.state, "syncing");
    assert.ok(after.every((bucket) => bucket.state === "synchronized"));
  });

  it("returns all events active at the current time", () => {
    const simulation = simulateCudaStreams({ ...baseConfig, mode: "dual-safe" });
    const time = simulation.events.find((event) => event.id === "allreduce-b2")!.start + 0.1;
    const active = activeCudaStreamEvents(simulation, time);

    assert.ok(active.some((event) => event.id === "allreduce-b2"));
    assert.ok(active.some((event) => event.kind === "compute"));
  });

  it("keeps a raced bucket invalid even after its producer later finishes", () => {
    const simulation = simulateCudaStreams({ ...baseConfig, mode: "dual-unsafe" });
    const bucketId = simulation.hazardBucketIds[0];
    const collective = simulation.events.find((event) => event.id === `allreduce-b${bucketId}`)!;
    const producer = simulation.events.find((event) => event.id === `backward-b${bucketId}`)!;
    const afterProducer = Math.min(
      collective.start + collective.duration - 0.01,
      producer.start + producer.duration + 0.01,
    );

    assert.equal(
      bucketSnapshotsAt(simulation, afterProducer).find((bucket) => bucket.bucketId === bucketId)?.state,
      "race",
    );
    assert.equal(
      bucketSnapshotsAt(simulation, collective.start + collective.duration).find((bucket) => bucket.bucketId === bucketId)?.state,
      "invalid",
    );
  });

  it("rejects unsupported bucket counts and communication durations", () => {
    assert.throws(() => simulateCudaStreams({ mode: "single", bucketCount: 1, communicationDuration: 2 }), /bucketCount/);
    assert.throws(() => simulateCudaStreams({ mode: "single", bucketCount: 3, communicationDuration: 0 }), /communicationDuration/);
  });
});
