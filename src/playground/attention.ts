export const ATTENTION_TOKENS = ["我", "喜欢", "机器", "学习"] as const;

export type AttentionMaskMode = "causal" | "bidirectional";
export type AttentionScaleMode = "scaled" | "unscaled";
export type AttentionPhaseId = "input" | "project" | "score" | "scale" | "mask" | "softmax" | "mix";

export interface AttentionPhase {
  id: AttentionPhaseId;
  compactLabel: string;
  label: string;
  formula: string;
  explanation: string;
}

export interface AttentionSimulation {
  tokens: string[];
  maskMode: AttentionMaskMode;
  scaleMode: AttentionScaleMode;
  headDim: number;
  valueDim: number;
  x: number[][];
  q: number[][];
  k: number[][];
  v: number[][];
  rawScores: number[][];
  scaledScores: number[][];
  maskedScores: Array<Array<number | null>>;
  weights: number[][];
  output: number[][];
}

export interface AttentionQueryDetail {
  queryIndex: number;
  keyIndex: number;
  rawScore: number;
  scaleDivisor: number;
  scoreAfterScale: number;
  masked: boolean;
  weight: number;
  keyVector: number[];
  valueVector: number[];
  weightedValue: number[];
  outputVector: number[];
}

export interface AttentionCost {
  sequenceLength: number;
  headCount: number;
  headDim: number;
  dtypeBytes: number;
  scoreElements: number;
  causalUsefulScoreElements: number;
  scoreBytes: number;
  qkvElements: number;
  qkvBytes: number;
  outputElements: number;
  outputBytes: number;
  qkFlops: number;
  avFlops: number;
  totalAttentionFlops: number;
}

export interface AttentionHeadLayout {
  modelDim: number;
  headCount: number;
  headDim: number;
  sequenceLength: number;
  projectedShape: [number, number];
  headShape: [number, number, number];
  scoreMatrices: number;
  scoreElements: number;
  concatenatedShape: [number, number];
}

export const ATTENTION_X: number[][] = [
  [1, 0, 1],
  [0, 1, 1],
  [1, 1, 0],
  [1, 1, 1],
];

export const ATTENTION_WQ: number[][] = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

export const ATTENTION_WK: number[][] = [
  [1, 0, 0],
  [0, 0, 1],
  [0, 1, 0],
];

export const ATTENTION_WV: number[][] = [
  [1, 0],
  [0, 1],
  [1, 1],
];

export const attentionPhases: AttentionPhase[] = [
  {
    id: "input",
    compactLabel: "Input X",
    label: "每个 token 先有一个输入向量",
    formula: "X ∈ ℝᵀˣᴰ",
    explanation: "Self-Attention 的 Q、K、V 都来自同一份序列 X，因此叫 self。不同投影让同一个 token 扮演三种角色。",
  },
  {
    id: "project",
    compactLabel: "Q / K / V",
    label: "三个 Linear 投影产生 Q、K、V",
    formula: "Q=XWq, K=XWk, V=XWv",
    explanation: "Wq、Wk、Wv 是训练得到的参数。Query 用来发问，Key 用来匹配，Value 携带最终要聚合的信息。",
  },
  {
    id: "score",
    compactLabel: "QKᵀ",
    label: "每个 Query 与所有 Key 做点积",
    formula: "S = QKᵀ",
    explanation: "一个 score cell 对应一对 token。整张 T×T 矩阵可以用一次矩阵乘法并行计算。",
  },
  {
    id: "scale",
    compactLabel: "Scale",
    label: "用 √dₖ 缩放点积",
    formula: "S' = S / √dₖ",
    explanation: "维度增大时点积幅度容易增大。缩放能让送入 Softmax 的值保持更温和，避免分布过早变得极端。",
  },
  {
    id: "mask",
    compactLabel: "Mask",
    label: "在 Softmax 前屏蔽不允许看到的位置",
    formula: "S' + attention_bias",
    explanation: "Causal mask 把未来位置设为负无穷，因此概率严格为 0。双向模式则允许每个 token 查看整段序列。",
  },
  {
    id: "softmax",
    compactLabel: "Softmax",
    label: "把每一行 score 变成和为 1 的权重",
    formula: "P = softmax(S')",
    explanation: "Softmax 沿 Key 方向逐行计算。每个 Query 得到自己的一组权重，而不是整张矩阵共享一组概率。",
  },
  {
    id: "mix",
    compactLabel: "PV",
    label: "用权重混合所有 Value",
    formula: "O = PV",
    explanation: "输出 token 仍然占据原位置，但向量已经吸收了其他可见 token 的信息。权重决定每个 Value 的贡献大小。",
  },
];

function assertRectangular(name: string, matrix: number[][]): void {
  const width = matrix[0]?.length ?? 0;
  if (matrix.length === 0 || width === 0 || matrix.some((row) => row.length !== width)) {
    throw new Error(`${name} must be a non-empty rectangular matrix`);
  }
  if (matrix.some((row) => row.some((value) => !Number.isFinite(value)))) {
    throw new Error(`${name} must contain finite values`);
  }
}

function assertIndex(name: string, value: number, size: number): void {
  if (!Number.isInteger(value) || value < 0 || value >= size) {
    throw new RangeError(`${name} is outside the sequence`);
  }
}

export function multiplyMatrices(left: number[][], right: number[][]): number[][] {
  assertRectangular("left", left);
  assertRectangular("right", right);
  if (left[0].length !== right.length) throw new Error("Matrix inner dimensions must match");

  return left.map((leftRow) =>
    Array.from({ length: right[0].length }, (_, column) =>
      leftRow.reduce((sum, value, inner) => sum + value * right[inner][column], 0),
    ),
  );
}

export function transposeMatrix(matrix: number[][]): number[][] {
  assertRectangular("matrix", matrix);
  return Array.from({ length: matrix[0].length }, (_, column) =>
    matrix.map((row) => row[column]),
  );
}

export function createAttentionSimulation(
  maskMode: AttentionMaskMode = "causal",
  scaleMode: AttentionScaleMode = "scaled",
): AttentionSimulation {
  const q = multiplyMatrices(ATTENTION_X, ATTENTION_WQ);
  const k = multiplyMatrices(ATTENTION_X, ATTENTION_WK);
  const v = multiplyMatrices(ATTENTION_X, ATTENTION_WV);
  const rawScores = multiplyMatrices(q, transposeMatrix(k));
  const headDim = q[0].length;
  const scaleDivisor = scaleMode === "scaled" ? Math.sqrt(headDim) : 1;
  const scaledScores = rawScores.map((row) => row.map((score) => score / scaleDivisor));
  const maskedScores = scaledScores.map((row, queryIndex) =>
    row.map((score, keyIndex) => maskMode === "causal" && keyIndex > queryIndex ? null : score),
  );
  const weights = maskedScores.map((row) => {
    const finiteScores = row.filter((score): score is number => score !== null);
    const maximum = Math.max(...finiteScores);
    const exponentials = row.map((score) => score === null ? 0 : Math.exp(score - maximum));
    const denominator = exponentials.reduce((sum, value) => sum + value, 0);
    return exponentials.map((value) => value / denominator);
  });
  const output = multiplyMatrices(weights, v);

  return {
    tokens: [...ATTENTION_TOKENS],
    maskMode,
    scaleMode,
    headDim,
    valueDim: v[0].length,
    x: ATTENTION_X.map((row) => [...row]),
    q,
    k,
    v,
    rawScores,
    scaledScores,
    maskedScores,
    weights,
    output,
  };
}

export function selectedAttentionDetail(
  simulation: AttentionSimulation,
  queryIndex: number,
  keyIndex: number,
): AttentionQueryDetail {
  assertIndex("queryIndex", queryIndex, simulation.tokens.length);
  assertIndex("keyIndex", keyIndex, simulation.tokens.length);
  const masked = simulation.maskedScores[queryIndex][keyIndex] === null;
  const weight = simulation.weights[queryIndex][keyIndex];

  return {
    queryIndex,
    keyIndex,
    rawScore: simulation.rawScores[queryIndex][keyIndex],
    scaleDivisor: simulation.scaleMode === "scaled" ? Math.sqrt(simulation.headDim) : 1,
    scoreAfterScale: simulation.scaledScores[queryIndex][keyIndex],
    masked,
    weight,
    keyVector: [...simulation.k[keyIndex]],
    valueVector: [...simulation.v[keyIndex]],
    weightedValue: simulation.v[keyIndex].map((value) => value * weight),
    outputVector: [...simulation.output[queryIndex]],
  };
}

export function attentionMatrixForPhase(
  simulation: AttentionSimulation,
  phase: AttentionPhaseId,
): Array<Array<number | null>> | null {
  if (phase === "score") return simulation.rawScores.map((row) => [...row]);
  if (phase === "scale") return simulation.scaledScores.map((row) => [...row]);
  if (phase === "mask") return simulation.maskedScores.map((row) => [...row]);
  if (phase === "softmax" || phase === "mix") return simulation.weights.map((row) => [...row]);
  return null;
}

export function attentionHeadLayout(
  modelDim: number,
  headCount: number,
  sequenceLength: number,
): AttentionHeadLayout {
  if (!Number.isInteger(modelDim) || modelDim <= 0) throw new RangeError("modelDim must be positive");
  if (!Number.isInteger(headCount) || headCount <= 0) throw new RangeError("headCount must be positive");
  if (!Number.isInteger(sequenceLength) || sequenceLength <= 0) throw new RangeError("sequenceLength must be positive");
  if (modelDim % headCount !== 0) throw new RangeError("modelDim must be divisible by headCount");
  const headDim = modelDim / headCount;

  return {
    modelDim,
    headCount,
    headDim,
    sequenceLength,
    projectedShape: [sequenceLength, modelDim],
    headShape: [headCount, sequenceLength, headDim],
    scoreMatrices: headCount,
    scoreElements: headCount * sequenceLength * sequenceLength,
    concatenatedShape: [sequenceLength, modelDim],
  };
}

export function attentionCost(
  sequenceLength: number,
  headCount: number,
  headDim: number,
  dtypeBytes = 2,
): AttentionCost {
  for (const [name, value] of [["sequenceLength", sequenceLength], ["headCount", headCount], ["headDim", headDim], ["dtypeBytes", dtypeBytes]] as const) {
    if (!Number.isInteger(value) || value <= 0) throw new RangeError(`${name} must be a positive integer`);
  }
  const scoreElements = headCount * sequenceLength * sequenceLength;
  const qkvElements = 3 * sequenceLength * headCount * headDim;
  const outputElements = sequenceLength * headCount * headDim;
  const qkFlops = 2 * scoreElements * headDim;
  const avFlops = 2 * scoreElements * headDim;

  return {
    sequenceLength,
    headCount,
    headDim,
    dtypeBytes,
    scoreElements,
    causalUsefulScoreElements: headCount * sequenceLength * (sequenceLength + 1) / 2,
    scoreBytes: scoreElements * dtypeBytes,
    qkvElements,
    qkvBytes: qkvElements * dtypeBytes,
    outputElements,
    outputBytes: outputElements * dtypeBytes,
    qkFlops,
    avFlops,
    totalAttentionFlops: qkFlops + avFlops,
  };
}
