import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { parseHash } from "./router";

describe("Atlas hash router", () => {
  it("routes the knowledge map and learning paths", () => {
    assert.deepEqual(parseHash(""), { kind: "home", section: "atlas" });
    assert.deepEqual(parseHash("#/"), { kind: "home", section: "atlas" });
    assert.deepEqual(parseHash("#/paths"), { kind: "home", section: "paths" });
  });

  it("routes all available distributed playgrounds as topics", () => {
    assert.deepEqual(parseHash("#/distributed/ddp"), { kind: "topic", topicId: "ddp" });
    assert.deepEqual(parseHash("#/distributed/zero-1"), { kind: "topic", topicId: "zero-1" });
    assert.deepEqual(parseHash("#/distributed/fsdp"), { kind: "topic", topicId: "fsdp" });
    assert.deepEqual(parseHash("#/distributed/compare"), { kind: "topic", topicId: "compare" });
  });

  it("routes the GPU execution-model playground", () => {
    assert.deepEqual(parseHash("#/gpu/architecture"), { kind: "topic", topicId: "gpu-architecture" });
    assert.deepEqual(parseHash("#/gpu/cuda-stream"), { kind: "topic", topicId: "cuda-stream" });
  });

  it("preserves unknown paths for the not-found view", () => {
    assert.deepEqual(parseHash("#/model/attention"), { kind: "not-found", path: "/model/attention" });
  });
});
