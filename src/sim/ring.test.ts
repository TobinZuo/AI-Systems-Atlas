import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { ddpGradientExample, simulateRingAllReduce } from "./ring";

describe("Ring All-Reduce simulation", () => {
  it("produces the expected element-wise SUM and average", () => {
    assert.deepEqual(ddpGradientExample.reduced, [
      1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888,
    ]);
    assert.deepEqual(ddpGradientExample.averaged, [
      277.75, 555.5, 833.25, 1111, 1388.75, 1666.5, 1944.25, 2222,
    ]);
  });

  it("uses N - 1 rounds for each phase", () => {
    assert.equal(ddpGradientExample.reduceScatter.length, 3);
    assert.equal(ddpGradientExample.allGather.length, 3);
    assert.ok(
      ddpGradientExample.reduceScatter[2].ranks.every(
        (rank) => rank.chunks.filter((chunk) => chunk.complete).length === 1,
      ),
    );
    assert.ok(
      ddpGradientExample.allGather[2].ranks.every((rank) =>
        rank.chunks.every((chunk) => chunk.complete),
      ),
    );
  });

  it("rejects tensors that cannot be split evenly around the ring", () => {
    assert.throws(
      () => simulateRingAllReduce([[1, 2, 3], [4, 5, 6]]),
      /divisible/,
    );
  });
});
