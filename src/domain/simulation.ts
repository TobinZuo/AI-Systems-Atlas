export type SystemLayer =
  | "framework"
  | "runtime"
  | "compute"
  | "memory"
  | "collective"
  | "interconnect"
  | "optimizer";

export type EventKind =
  | "launch"
  | "compute"
  | "write"
  | "ready"
  | "reduce-scatter"
  | "all-gather"
  | "synchronize"
  | "update";

export type ComponentId =
  | "python"
  | "autograd"
  | "cpu"
  | "gpu-0"
  | "gpu-1"
  | "gpu-2"
  | "gpu-3"
  | "sm"
  | "hbm"
  | "ddp"
  | "nccl"
  | "nvlink"
  | "optimizer";

export interface SimulationEvent {
  id: string;
  title: string;
  compactTitle: string;
  start: number;
  duration: number;
  kind: EventKind;
  layer: SystemLayer;
  actor: ComponentId;
  location: string;
  tensor?: string;
  source?: ComponentId;
  destination?: ComponentId;
  dependencies?: string[];
  explanation: string;
  details: string[];
}

export interface Scenario {
  id: string;
  title: string;
  subtitle: string;
  worldSize: number;
  totalDuration: number;
  events: SimulationEvent[];
}

export interface ComponentDefinition {
  id: ComponentId;
  name: string;
  category: string;
  role: string;
  knows: string;
  doesNotKnow: string;
}
