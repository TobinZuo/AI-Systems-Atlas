export type GradientReduction = "mean" | "sum";
export type GradientDevice = "cpu" | "cuda";
export type GradientDirection = "descent" | "ascent";
export type GradientBatchSize = 1 | 2 | 4;
export type GradientRankSplit = "2+2" | "1+3";
export type GradientParameter = "w0" | "w1" | "b";

export interface GradientConfig {
  w0: number;
  w1: number;
  b: number;
  batchSize: GradientBatchSize;
  reduction: GradientReduction;
  learningRate: number;
  direction: GradientDirection;
  device: GradientDevice;
}

export interface GradientVector {
  w0: number;
  w1: number;
  b: number;
}

export interface GradientSample {
  id: number;
  x: [number, number];
  target: number;
  prediction: number;
  residual: number;
  loss: number;
  gradient: GradientVector;
}

export interface GradientBuffer {
  name: "weight.grad" | "bias.grad";
  shape: "[2]" | "[1]";
  dtype: "float32";
  device: GradientDevice;
  memory: "CPU DRAM" | "GPU HBM";
  values: number[];
  bytes: number;
}

export interface FiniteDifferenceCheck {
  parameter: GradientParameter;
  analytic: number;
  numeric: number;
  absoluteError: number;
}

export interface GradientStepResult {
  delta: GradientVector;
  nextParameters: GradientVector;
  nextLoss: number;
  lossChange: number;
  predictedFirstOrderChange: number;
}

export interface GradientSimulation {
  config: GradientConfig;
  samples: GradientSample[];
  aggregateLoss: number;
  aggregateGradient: GradientVector;
  gradientNorm: number;
  buffers: GradientBuffer[];
  flattenedBucket: number[];
  step: GradientStepResult;
  finiteDifferences: FiniteDifferenceCheck[];
}

export interface GradientRankContribution {
  rank: number;
  sampleIds: number[];
  localMean: GradientVector;
  localBatchSize: number;
}

export interface GradientDDPBridge {
  split: GradientRankSplit;
  ranks: GradientRankContribution[];
  rankMean: GradientVector;
  sampleWeightedMean: GradientVector;
  globalBatchMean: GradientVector;
  rankMeanMatchesGlobal: boolean;
}

export const gradientDataset: ReadonlyArray<{
  id: number;
  x: readonly [number, number];
  target: number;
}> = [
  { id: 0, x: [1, 2], target: 0 },
  { id: 1, x: [2, -1], target: 3 },
  { id: 2, x: [-1, 1], target: -2 },
  { id: 3, x: [3, 1], target: 1 },
];

export const defaultGradientConfig: GradientConfig = {
  w0: 1,
  w1: -1,
  b: 0.5,
  batchSize: 4,
  reduction: "mean",
  learningRate: 0.1,
  direction: "descent",
  device: "cuda",
};

const vectorParameters: GradientParameter[] = ["w0", "w1", "b"];

function add(a: GradientVector, b: GradientVector): GradientVector {
  return { w0: a.w0 + b.w0, w1: a.w1 + b.w1, b: a.b + b.b };
}

function scale(vector: GradientVector, factor: number): GradientVector {
  return { w0: vector.w0 * factor, w1: vector.w1 * factor, b: vector.b * factor };
}

function mean(vectors: GradientVector[]): GradientVector {
  if (vectors.length === 0) throw new Error("cannot average an empty gradient list");
  return scale(vectors.reduce(add, { w0: 0, w1: 0, b: 0 }), 1 / vectors.length);
}

function evaluateSample(
  definition: (typeof gradientDataset)[number],
  parameters: GradientVector,
): GradientSample {
  const prediction = parameters.w0 * definition.x[0] + parameters.w1 * definition.x[1] + parameters.b;
  const residual = prediction - definition.target;
  return {
    id: definition.id,
    x: [...definition.x],
    target: definition.target,
    prediction,
    residual,
    loss: 0.5 * residual ** 2,
    gradient: {
      w0: residual * definition.x[0],
      w1: residual * definition.x[1],
      b: residual,
    },
  };
}

function validateConfig(config: GradientConfig): void {
  const values = [config.w0, config.w1, config.b, config.learningRate];
  if (values.some((value) => !Number.isFinite(value))) throw new Error("gradient config values must be finite");
  if (![1, 2, 4].includes(config.batchSize)) throw new Error("batchSize must be 1, 2, or 4");
  if (config.learningRate <= 0) throw new Error("learningRate must be positive");
  if (!(["mean", "sum"] as string[]).includes(config.reduction)) throw new Error("unsupported reduction");
  if (!(["descent", "ascent"] as string[]).includes(config.direction)) throw new Error("unsupported direction");
  if (!(["cpu", "cuda"] as string[]).includes(config.device)) throw new Error("unsupported device");
}

function aggregateSamples(samples: GradientSample[], reduction: GradientReduction): {
  loss: number;
  gradient: GradientVector;
} {
  const lossSum = samples.reduce((sum, sample) => sum + sample.loss, 0);
  const gradientSum = samples.map((sample) => sample.gradient).reduce(add, { w0: 0, w1: 0, b: 0 });
  const factor = reduction === "mean" ? 1 / samples.length : 1;
  return { loss: lossSum * factor, gradient: scale(gradientSum, factor) };
}

function objective(config: GradientConfig, parameters: GradientVector): number {
  const samples = gradientDataset
    .slice(0, config.batchSize)
    .map((definition) => evaluateSample(definition, parameters));
  return aggregateSamples(samples, config.reduction).loss;
}

export function simulateGradient(config: GradientConfig): GradientSimulation {
  validateConfig(config);
  const parameters = { w0: config.w0, w1: config.w1, b: config.b };
  const samples = gradientDataset
    .slice(0, config.batchSize)
    .map((definition) => evaluateSample(definition, parameters));
  const aggregate = aggregateSamples(samples, config.reduction);
  const gradientNorm = Math.hypot(aggregate.gradient.w0, aggregate.gradient.w1, aggregate.gradient.b);
  const sign = config.direction === "descent" ? -1 : 1;
  const delta = scale(aggregate.gradient, sign * config.learningRate);
  const nextParameters = add(parameters, delta);
  const nextLoss = objective(config, nextParameters);
  const predictedFirstOrderChange =
    aggregate.gradient.w0 * delta.w0
    + aggregate.gradient.w1 * delta.w1
    + aggregate.gradient.b * delta.b;
  const epsilon = 1e-4;
  const finiteDifferences = vectorParameters.map((parameter): FiniteDifferenceCheck => {
    const positive = { ...parameters, [parameter]: parameters[parameter] + epsilon };
    const negative = { ...parameters, [parameter]: parameters[parameter] - epsilon };
    const numeric = (objective(config, positive) - objective(config, negative)) / (2 * epsilon);
    return {
      parameter,
      analytic: aggregate.gradient[parameter],
      numeric,
      absoluteError: Math.abs(aggregate.gradient[parameter] - numeric),
    };
  });
  const memory = config.device === "cuda" ? "GPU HBM" : "CPU DRAM";

  return {
    config,
    samples,
    aggregateLoss: aggregate.loss,
    aggregateGradient: aggregate.gradient,
    gradientNorm,
    buffers: [
      { name: "weight.grad", shape: "[2]", dtype: "float32", device: config.device, memory, values: [aggregate.gradient.w0, aggregate.gradient.w1], bytes: 8 },
      { name: "bias.grad", shape: "[1]", dtype: "float32", device: config.device, memory, values: [aggregate.gradient.b], bytes: 4 },
    ],
    flattenedBucket: [aggregate.gradient.w0, aggregate.gradient.w1, aggregate.gradient.b],
    step: {
      delta,
      nextParameters,
      nextLoss,
      lossChange: nextLoss - aggregate.loss,
      predictedFirstOrderChange,
    },
    finiteDifferences,
  };
}

export function simulateGradientDDP(
  parameters: Pick<GradientConfig, "w0" | "w1" | "b">,
  split: GradientRankSplit,
): GradientDDPBridge {
  const sampleGroups = split === "2+2" ? [[0, 1], [2, 3]] : [[0], [1, 2, 3]];
  const rankContributions = sampleGroups.map((sampleIds, rank): GradientRankContribution => {
    const gradients = sampleIds.map((sampleId) => evaluateSample(gradientDataset[sampleId], parameters).gradient);
    return { rank, sampleIds, localMean: mean(gradients), localBatchSize: sampleIds.length };
  });
  const rankMean = mean(rankContributions.map((rank) => rank.localMean));
  const totalSamples = rankContributions.reduce((sum, rank) => sum + rank.localBatchSize, 0);
  const sampleWeightedMean = scale(
    rankContributions
      .map((rank) => scale(rank.localMean, rank.localBatchSize))
      .reduce(add, { w0: 0, w1: 0, b: 0 }),
    1 / totalSamples,
  );
  const globalBatchMean = mean(
    gradientDataset.map((definition) => evaluateSample(definition, parameters).gradient),
  );
  const rankMeanMatchesGlobal = vectorParameters.every(
    (parameter) => Math.abs(rankMean[parameter] - globalBatchMean[parameter]) < 1e-10,
  );

  return {
    split,
    ranks: rankContributions,
    rankMean,
    sampleWeightedMean,
    globalBatchMean,
    rankMeanMatchesGlobal,
  };
}
