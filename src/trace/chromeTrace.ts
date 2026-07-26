import type {
  TraceCategory,
  TraceDataset,
  TraceEvent,
  TraceImportResult,
  TraceLane,
} from "../domain/trace";

interface ChromeTraceRecord {
  name?: unknown;
  cat?: unknown;
  ph?: unknown;
  ts?: unknown;
  dur?: unknown;
  pid?: unknown;
  tid?: unknown;
  args?: unknown;
}

interface OpenSlice {
  record: ChromeTraceRecord;
  start: number;
}

const asNumber = (value: unknown) =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

const asText = (value: unknown, fallback: string) =>
  typeof value === "string" && value.trim() ? value : fallback;

const asArgs = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};

const laneId = (record: ChromeTraceRecord) =>
  `${asText(record.pid, String(record.pid ?? "0"))}:${asText(record.tid, String(record.tid ?? "0"))}`;

export function inferTraceCategory(name: string, category = ""): TraceCategory {
  const text = `${name} ${category}`.toLowerCase();
  if (/nccl|allreduce|all-reduce|all_gather|reduce.?scatter|collective/.test(text)) return "collective";
  if (/nvlink|pcie|rdma|network|send|recv|memcpy p2p/.test(text)) return "network";
  if (/hbm|memory|allocator|memcpy|memset|dram/.test(text)) return "memory";
  if (/cuda|kernel|gemm|matmul|conv|adam|optimizer|gpu/.test(text)) return "compute";
  if (/python|autograd|backward|forward|dataloader|framework/.test(text)) return "framework";
  if (/cpu|thread|runtime|operator|aten/.test(text)) return "cpu";
  return "other";
}

export function parseChromeTrace(input: unknown, fileName = "imported-trace.json"): TraceImportResult {
  const root = input && typeof input === "object" ? input as Record<string, unknown> : null;
  const rawRecords = Array.isArray(root?.traceEvents)
    ? root.traceEvents
    : Array.isArray(input)
      ? input
      : null;
  if (!rawRecords) {
    throw new Error("找不到 traceEvents。请选择 Chrome/PyTorch Profiler 导出的 JSON 文件。");
  }

  const records = rawRecords.filter(
    (record): record is ChromeTraceRecord => Boolean(record && typeof record === "object"),
  );
  const threadNames = new Map<string, string>();
  const processNames = new Map<string, string>();

  for (const record of records) {
    if (record.ph !== "M") continue;
    const args = asArgs(record.args);
    const name = asText(args.name, "");
    const pid = String(record.pid ?? "0");
    if (record.name === "thread_name" && name) threadNames.set(laneId(record), name);
    if (record.name === "process_name" && name) processNames.set(pid, name);
  }

  const complete: Array<{ record: ChromeTraceRecord; start: number; duration: number }> = [];
  const stacks = new Map<string, OpenSlice[]>();
  let skippedEvents = 0;

  for (const record of records) {
    const timestamp = asNumber(record.ts);
    const phase = record.ph;
    if (phase === "X" && timestamp !== null) {
      const duration = asNumber(record.dur);
      if (duration === null || duration < 0) {
        skippedEvents += 1;
      } else {
        complete.push({ record, start: timestamp, duration });
      }
      continue;
    }
    if (phase === "B" && timestamp !== null) {
      const key = laneId(record);
      const stack = stacks.get(key) ?? [];
      stack.push({ record, start: timestamp });
      stacks.set(key, stack);
      continue;
    }
    if (phase === "E" && timestamp !== null) {
      const key = laneId(record);
      const open = stacks.get(key)?.pop();
      if (open) complete.push({ record: open.record, start: open.start, duration: Math.max(0, timestamp - open.start) });
      else skippedEvents += 1;
      continue;
    }
    if (phase !== "M") skippedEvents += 1;
  }

  if (!complete.length) {
    throw new Error("文件中没有可显示的 Complete(X) 或 Begin/End(B/E) 事件。");
  }

  const minTimestamp = Math.min(...complete.map((slice) => slice.start));
  const laneMap = new Map<string, TraceLane>();
  const events: TraceEvent[] = complete.map((slice, index) => {
    const id = laneId(slice.record);
    const pid = String(slice.record.pid ?? "0");
    if (!laneMap.has(id)) {
      laneMap.set(id, {
        id,
        label: threadNames.get(id) ?? `Thread ${String(slice.record.tid ?? "0")}`,
        group: processNames.get(pid) ?? `Process ${pid}`,
        detail: `pid ${pid} / tid ${String(slice.record.tid ?? "0")}`,
      });
    }
    const name = asText(slice.record.name, "unnamed event");
    const category = asText(slice.record.cat, "");
    return {
      id: `imported-${index}`,
      name,
      laneId: id,
      start: (slice.start - minTimestamp) / 1000,
      duration: Math.max(slice.duration / 1000, 0.001),
      category: inferTraceCategory(name, category),
      args: {
        ...asArgs(slice.record.args),
        pid: slice.record.pid ?? 0,
        tid: slice.record.tid ?? 0,
        category,
        originalTimestampUs: slice.start,
      },
    };
  });

  const totalDuration = Math.max(...events.map((event) => event.start + event.duration));
  const warnings: string[] = [];
  if (skippedEvents) warnings.push(`跳过 ${skippedEvents} 个不支持或不完整的事件`);

  return {
    dataset: {
      id: `import-${Date.now()}`,
      name: fileName,
      description: "从 Chrome Trace Event JSON 导入，时间已归一化为毫秒。",
      source: "chrome-trace",
      timeUnit: "ms",
      totalDuration,
      lanes: [...laneMap.values()],
      events: events.sort((a, b) => a.start - b.start),
      warnings,
      metadata: {
        processes: processNames.size,
        threads: laneMap.size,
        originalEvents: records.length,
      },
    },
    importedEvents: events.length,
    skippedEvents,
  };
}
