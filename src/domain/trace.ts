export type TraceCategory =
  | "framework"
  | "cpu"
  | "compute"
  | "collective"
  | "memory"
  | "network"
  | "other";

export type TraceSource = "sample" | "chrome-trace";

export interface TraceLane {
  id: string;
  label: string;
  group: string;
  detail: string;
  rank?: number;
}

export interface TraceEvent {
  id: string;
  name: string;
  laneId: string;
  start: number;
  duration: number;
  category: TraceCategory;
  args: Record<string, unknown>;
  conceptEventId?: string;
}

export interface TraceDataset {
  id: string;
  name: string;
  description: string;
  source: TraceSource;
  timeUnit: "ms";
  totalDuration: number;
  lanes: TraceLane[];
  events: TraceEvent[];
  warnings: string[];
  metadata: Record<string, string | number>;
}

export interface TraceImportResult {
  dataset: TraceDataset;
  importedEvents: number;
  skippedEvents: number;
}
