export type AtlasRoute =
  | { kind: "home"; section: "atlas" | "paths" }
  | { kind: "topic"; topicId: "ddp" }
  | { kind: "not-found"; path: string };

export function parseHash(hash: string): AtlasRoute {
  const path = hash.replace(/^#/, "").replace(/\/+$/, "") || "/";

  if (path === "/" || path === "/atlas") return { kind: "home", section: "atlas" };
  if (path === "/paths") return { kind: "home", section: "paths" };
  if (path === "/distributed/ddp") return { kind: "topic", topicId: "ddp" };

  return { kind: "not-found", path };
}
