export type AtlasRoute =
  | { kind: "home"; section: "atlas" | "paths" }
  | { kind: "topic"; topicId: "autograd" | "gradient" | "adamw" | "attention" | "gpu-architecture" | "cuda-kernel" | "cuda-stream" | "memory-hierarchy" | "process-rank" | "collective" | "ring-allreduce" | "ddp" | "zero-1" | "fsdp" | "compare" }
  | { kind: "not-found"; path: string };

export function parseHash(hash: string): AtlasRoute {
  const path = hash.replace(/^#/, "").replace(/\/+$/, "") || "/";

  if (path === "/" || path === "/atlas") return { kind: "home", section: "atlas" };
  if (path === "/paths") return { kind: "home", section: "paths" };
  if (path === "/training/autograd") return { kind: "topic", topicId: "autograd" };
  if (path === "/training/gradient") return { kind: "topic", topicId: "gradient" };
  if (path === "/training/adamw") return { kind: "topic", topicId: "adamw" };
  if (path === "/model/attention") return { kind: "topic", topicId: "attention" };
  if (path === "/gpu/architecture") return { kind: "topic", topicId: "gpu-architecture" };
  if (path === "/gpu/cuda-kernel") return { kind: "topic", topicId: "cuda-kernel" };
  if (path === "/gpu/cuda-stream") return { kind: "topic", topicId: "cuda-stream" };
  if (path === "/gpu/memory-hierarchy") return { kind: "topic", topicId: "memory-hierarchy" };
  if (path === "/distributed/process-rank") return { kind: "topic", topicId: "process-rank" };
  if (path === "/distributed/collective") return { kind: "topic", topicId: "collective" };
  if (path === "/distributed/ring-allreduce") return { kind: "topic", topicId: "ring-allreduce" };
  if (path === "/distributed/ddp") return { kind: "topic", topicId: "ddp" };
  if (path === "/distributed/zero-1") return { kind: "topic", topicId: "zero-1" };
  if (path === "/distributed/fsdp") return { kind: "topic", topicId: "fsdp" };
  if (path === "/distributed/compare") return { kind: "topic", topicId: "compare" };

  return { kind: "not-found", path };
}
