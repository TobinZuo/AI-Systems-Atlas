import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { getAvailableScenario, scenarioRegistry } from "./registry";

describe("Scenario registry", () => {
  it("keeps scenario identifiers unique", () => {
    const ids = scenarioRegistry.map((scenario) => scenario.id);
    assert.equal(new Set(ids).size, ids.length);
  });

  it("gives every available topic page a unique route", () => {
    const routes = scenarioRegistry
      .filter((registration) => registration.status === "available")
      .map((registration) => registration.route);
    assert.ok(routes.every((route) => route));
    assert.equal(new Set(routes).size, routes.length);
  });

  it("only resolves implemented scenarios", () => {
    assert.equal(getAvailableScenario("ddp-gradient-journey").worldSize, 4);
    assert.throws(() => getAvailableScenario("fsdp-zero-3"), /not available/);
  });
});
