import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { parseHash } from "./router";

describe("Atlas hash router", () => {
  it("routes the knowledge map and learning paths", () => {
    assert.deepEqual(parseHash(""), { kind: "home", section: "atlas" });
    assert.deepEqual(parseHash("#/"), { kind: "home", section: "atlas" });
    assert.deepEqual(parseHash("#/paths"), { kind: "home", section: "paths" });
  });

  it("routes the existing DDP playground as a topic", () => {
    assert.deepEqual(parseHash("#/distributed/ddp"), { kind: "topic", topicId: "ddp" });
  });

  it("preserves unknown paths for the not-found view", () => {
    assert.deepEqual(parseHash("#/model/attention"), { kind: "not-found", path: "/model/attention" });
  });
});
