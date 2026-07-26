export type AdamWMode = "adamw" | "coupled-l2";
export type AdamWParameterId = "weight-0" | "weight-1" | "bias";
export type AdamWMemoryStrategy = "ddp" | "zero-1" | "fsdp";

export interface AdamWConfig {
  beta1: number;
  beta2: number;
  epsilon: number;
  learningRate: number;
  weightDecay: number;
  mode: AdamWMode;
}

export interface AdamWParameterDefinition {
  id: AdamWParameterId;
  label: string;
  tensorName: "weight" | "bias";
  group: "decay" | "no-decay";
  initialValue: number;
  decayApplied: boolean;
}

export interface AdamWParameterTrace {
  id: AdamWParameterId;
  label: string;
  tensorName: "weight" | "bias";
  group: "decay" | "no-decay";
  step: number;
  gradient: number;
  parameterBefore: number;
  momentInput: number;
  expAvgBefore: number;
  expAvg: number;
  expAvgSqBefore: number;
  expAvgSq: number;
  biasCorrection1: number;
  biasCorrection2: number;
  correctedAvg: number;
  correctedAvgSq: number;
  adaptiveDirection: number;
  gradientUpdate: number;
  decayUpdate: number;
  parameterAfter: number;
  decayApplied: boolean;
}

export interface AdamWStepTrace {
  step: number;
  gradients: Record<AdamWParameterId, number>;
  parameters: AdamWParameterTrace[];
}

export interface AdamWSimulation {
  config: AdamWConfig;
  steps: AdamWStepTrace[];
}

export interface AdamWMemoryBreakdown {
  strategy: AdamWMemoryStrategy;
  worldSize: number;
  parameterBytes: number;
  gradientBytes: number;
  firstMomentBytes: number;
  secondMomentBytes: number;
  persistentBytesPerRank: number;
  replicatedComponents: string[];
  shardedComponents: string[];
}

export const adamWParameters: AdamWParameterDefinition[] = [
  { id: "weight-0", label: "weight[0]", tensorName: "weight", group: "decay", initialValue: 1, decayApplied: true },
  { id: "weight-1", label: "weight[1]", tensorName: "weight", group: "decay", initialValue: -1, decayApplied: true },
  { id: "bias", label: "bias", tensorName: "bias", group: "no-decay", initialValue: 0.5, decayApplied: false },
];

export const adamWGradientSchedule: ReadonlyArray<Record<AdamWParameterId, number>> = [
  { "weight-0": 1.125, "weight-1": 0.125, bias: 0.5 },
  { "weight-0": 0.8, "weight-1": -0.2, bias: 0.35 },
  { "weight-0": -0.4, "weight-1": 0.6, bias: 0.1 },
  { "weight-0": -0.8, "weight-1": 0.3, bias: -0.2 },
  { "weight-0": 0.2, "weight-1": -0.5, bias: -0.1 },
];

export const defaultAdamWConfig: AdamWConfig = {
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
  learningRate: 0.1,
  weightDecay: 0.1,
  mode: "adamw",
};

function validateConfig(config: AdamWConfig): void {
  const values = [config.beta1, config.beta2, config.epsilon, config.learningRate, config.weightDecay];
  if (values.some((value) => !Number.isFinite(value))) throw new Error("AdamW config values must be finite");
  if (config.beta1 < 0 || config.beta1 >= 1) throw new Error("beta1 must be in [0, 1)");
  if (config.beta2 < 0 || config.beta2 >= 1) throw new Error("beta2 must be in [0, 1)");
  if (config.epsilon <= 0) throw new Error("epsilon must be positive");
  if (config.learningRate <= 0) throw new Error("learningRate must be positive");
  if (config.weightDecay < 0) throw new Error("weightDecay cannot be negative");
  if (config.mode !== "adamw" && config.mode !== "coupled-l2") throw new Error("unsupported optimizer mode");
}

export function simulateAdamW(config: AdamWConfig): AdamWSimulation {
  validateConfig(config);

  const state = new Map(
    adamWParameters.map((parameter) => [parameter.id, {
      parameter: parameter.initialValue,
      expAvg: 0,
      expAvgSq: 0,
    }]),
  );

  const steps = adamWGradientSchedule.map((gradients, scheduleIndex): AdamWStepTrace => {
    const step = scheduleIndex + 1;
    const parameters = adamWParameters.map((definition): AdamWParameterTrace => {
      const previous = state.get(definition.id);
      if (!previous) throw new Error(`missing optimizer state for ${definition.id}`);

      const gradient = gradients[definition.id];
      const coupledPenalty = definition.decayApplied && config.mode === "coupled-l2"
        ? config.weightDecay * previous.parameter
        : 0;
      const momentInput = gradient + coupledPenalty;
      const expAvg = config.beta1 * previous.expAvg + (1 - config.beta1) * momentInput;
      const expAvgSq = config.beta2 * previous.expAvgSq + (1 - config.beta2) * momentInput ** 2;
      const biasCorrection1 = 1 - config.beta1 ** step;
      const biasCorrection2 = 1 - config.beta2 ** step;
      const correctedAvg = expAvg / biasCorrection1;
      const correctedAvgSq = expAvgSq / biasCorrection2;
      const adaptiveDirection = correctedAvg / (Math.sqrt(correctedAvgSq) + config.epsilon);
      const gradientUpdate = -config.learningRate * adaptiveDirection;
      const decayUpdate = definition.decayApplied && config.mode === "adamw"
        ? -config.learningRate * config.weightDecay * previous.parameter
        : 0;
      const parameterAfter = previous.parameter + decayUpdate + gradientUpdate;

      state.set(definition.id, { parameter: parameterAfter, expAvg, expAvgSq });
      return {
        id: definition.id,
        label: definition.label,
        tensorName: definition.tensorName,
        group: definition.group,
        step,
        gradient,
        parameterBefore: previous.parameter,
        momentInput,
        expAvgBefore: previous.expAvg,
        expAvg,
        expAvgSqBefore: previous.expAvgSq,
        expAvgSq,
        biasCorrection1,
        biasCorrection2,
        correctedAvg,
        correctedAvgSq,
        adaptiveDirection,
        gradientUpdate,
        decayUpdate,
        parameterAfter,
        decayApplied: definition.decayApplied,
      };
    });

    return { step, gradients, parameters };
  });

  return { config, steps };
}

export function getAdamWMemoryBreakdown(
  strategy: AdamWMemoryStrategy,
  parameterCount: number,
  worldSize: number,
): AdamWMemoryBreakdown {
  if (!Number.isInteger(parameterCount) || parameterCount < 1) throw new Error("parameterCount must be positive");
  if (!Number.isInteger(worldSize) || worldSize < 1) throw new Error("worldSize must be positive");

  const tensorBytes = parameterCount * 4;
  const shard = (bytes: number) => bytes / worldSize;
  const parameterBytes = strategy === "fsdp" ? shard(tensorBytes) : tensorBytes;
  const gradientBytes = strategy === "fsdp" ? shard(tensorBytes) : tensorBytes;
  const firstMomentBytes = strategy === "ddp" ? tensorBytes : shard(tensorBytes);
  const secondMomentBytes = strategy === "ddp" ? tensorBytes : shard(tensorBytes);

  return {
    strategy,
    worldSize,
    parameterBytes,
    gradientBytes,
    firstMomentBytes,
    secondMomentBytes,
    persistentBytesPerRank: parameterBytes + gradientBytes + firstMomentBytes + secondMomentBytes,
    replicatedComponents: strategy === "ddp" ? ["parameter", "gradient", "m", "v"] : strategy === "zero-1" ? ["parameter", "gradient"] : [],
    shardedComponents: strategy === "ddp" ? [] : strategy === "zero-1" ? ["m", "v"] : ["parameter", "gradient", "m", "v"],
  };
}
