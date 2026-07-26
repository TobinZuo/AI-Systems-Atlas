import type { Scenario } from "../domain/simulation";
import { ddpScenario } from "./ddp";

export interface ScenarioRegistration {
  id: string;
  title: string;
  description: string;
  domain: "distributed";
  route?: string;
  evidence: string[];
  status: "available" | "planned";
  capabilities: string[];
  scenario?: Scenario;
}

export const scenarioRegistry: ScenarioRegistration[] = [
  {
    id: ddpScenario.id,
    title: "DDP：一个梯度的旅程",
    description: "从 backward、GPU kernel、Ring All-Reduce 到 AdamW。",
    domain: "distributed",
    route: "/distributed/ddp",
    evidence: ["CS336 Assignment 2", "deterministic simulator", "Chrome Trace"],
    status: "available",
    capabilities: ["concept", "numeric-state", "direct-manipulation", "trace-sample"],
    scenario: ddpScenario,
  },
  {
    id: "zero-1-sharded-optimizer",
    title: "ZeRO-1：Sharded Optimizer",
    description: "Optimizer state 分片、owner 更新与参数广播。",
    domain: "distributed",
    route: "/distributed/zero-1",
    evidence: ["CS336 Assignment 2", "smoke test"],
    status: "available",
    capabilities: ["concept", "numeric-state", "direct-manipulation", "comparison"],
  },
  {
    id: "fsdp-zero-3",
    title: "FSDP：参数按需重建",
    description: "All-Gather 参数、Reduce-Scatter 梯度与显存生命周期。",
    domain: "distributed",
    route: "/distributed/fsdp",
    evidence: ["CS336 Assignment 2", "FSDP implementation"],
    status: "available",
    capabilities: ["concept", "numeric-state", "direct-manipulation", "comparison", "memory-lifetime"],
  },
];

export function getAvailableScenario(id: string): Scenario {
  const registration = scenarioRegistry.find((item) => item.id === id);
  if (!registration?.scenario) throw new Error(`Scenario ${id} is not available`);
  return registration.scenario;
}
