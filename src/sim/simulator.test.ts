import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { ddpScenario } from "../scenarios/ddp";
import {
  eventAtTime,
  eventProgress,
  nextEventTime,
  previousEventTime,
  validateScenario,
} from "./simulator";

describe("DDP simulation timeline", () => {
  it("contains a valid dependency-ordered scenario", () => {
    assert.deepEqual(validateScenario(ddpScenario), []);
  });

  it("selects the active event at a given time", () => {
    assert.equal(eventAtTime(ddpScenario.events, 0).id, "python-backward");
    assert.equal(
      eventAtTime(ddpScenario.events, 6.7).id,
      "reduce-scatter-0",
    );
    assert.equal(
      eventAtTime(ddpScenario.events, 16.9).id,
      "iteration-complete",
    );
  });

  it("clamps event progress to the zero-to-one interval", () => {
    const event = ddpScenario.events[0];
    assert.equal(eventProgress(event, -10), 0);
    assert.ok(Math.abs(eventProgress(event, 0.6) - 0.5) < 0.0001);
    assert.equal(eventProgress(event, 10), 1);
  });

  it("steps between event boundaries", () => {
    assert.equal(nextEventTime(ddpScenario.events, 0), 1.2);
    assert.equal(previousEventTime(ddpScenario.events, 7.8), 6.6);
  });
});
