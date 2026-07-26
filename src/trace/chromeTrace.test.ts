import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { inferTraceCategory, parseChromeTrace } from "./chromeTrace";

describe("Chrome Trace importer", () => {
  it("imports complete and begin/end events into normalized lanes", () => {
    const result = parseChromeTrace({
      traceEvents: [
        { ph: "M", name: "process_name", pid: 7, tid: 0, args: { name: "rank 0" } },
        { ph: "M", name: "thread_name", pid: 7, tid: 2, args: { name: "CUDA stream 7" } },
        { ph: "X", name: "gemm_backward", cat: "cuda", pid: 7, tid: 2, ts: 1000, dur: 500, args: { shape: "8x8" } },
        { ph: "B", name: "ncclAllReduce", cat: "collective", pid: 7, tid: 3, ts: 1600 },
        { ph: "E", pid: 7, tid: 3, ts: 2200 },
      ],
    });

    assert.equal(result.importedEvents, 2);
    assert.equal(result.dataset.lanes.length, 2);
    assert.equal(result.dataset.events[0].start, 0);
    assert.equal(result.dataset.events[0].duration, 0.5);
    assert.equal(result.dataset.events[1].category, "collective");
    assert.equal(result.dataset.lanes[0].label, "CUDA stream 7");
  });

  it("rejects JSON without displayable trace events", () => {
    assert.throws(() => parseChromeTrace({ hello: "world" }), /traceEvents/);
    assert.throws(() => parseChromeTrace({ traceEvents: [] }), /没有可显示/);
  });

  it("recognizes common AI systems event categories", () => {
    assert.equal(inferTraceCategory("ncclKernel_AllReduce"), "collective");
    assert.equal(inferTraceCategory("NVLink P2P send"), "network");
    assert.equal(inferTraceCategory("gemm_backward"), "compute");
  });
});
