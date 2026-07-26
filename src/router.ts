export type AtlasRoute =
  | { kind: "home"; section: "atlas" | "paths" }
  | { kind: "topic"; topicId: "ddp" | "zero-1" | "fsdp" | "compare" }
  | { kind: "not-found"; path: string };

export function parseHash(hash: string): AtlasRoute {
  const path = hash.replace(/^#/, "").replace(/\/+$/, "") || "/";

  if (path === "/" || path === "/atlas") return { kind: "home", section: "atlas" };
  if (path === "/paths") return { kind: "home", section: "paths" };
  if (path === "/distributed/ddp") return { kind: "topic", topicId: "ddp" };
  if (path === "/distributed/zero-1") return { kind: "topic", topicId: "zero-1" };
  if (path === "/distributed/fsdp") return { kind: "topic", topicId: "fsdp" };
  if (path === "/distributed/compare") return { kind: "topic", topicId: "compare" };

  return { kind: "not-found", path };
}
