export const MEMORY_MATRIX_SIZE = 4;
export const MEMORY_FLOAT_BYTES = 4;

export type MemoryKernelStrategy = "naive" | "tiled";
export type MemoryMatrixId = "X" | "dY" | "dW";
export type MemoryStepId =
  | "global-request"
  | "l2-lookup"
  | "shared-stage"
  | "register-load"
  | "fma"
  | "writeback";

export interface MemoryStep {
  id: MemoryStepId;
  compactLabel: string;
  label: string;
  activeLayer: "hbm" | "l2" | "shared" | "register" | "compute";
  explanation: string;
}

export interface MatrixValue {
  id: string;
  matrix: MemoryMatrixId;
  row: number;
  column: number;
  value: number;
  selectedOperand: boolean;
}

export interface GradientEquation {
  row: number;
  column: number;
  terms: Array<{
    k: number;
    left: number;
    right: number;
    product: number;
  }>;
  result: number;
}

export interface GradientTile {
  tileSize: number;
  outputRowStart: number;
  outputColumnStart: number;
  kStart: number;
  outputRows: number[];
  outputColumns: number[];
  kValues: number[];
  xValues: MatrixValue[];
  dyValues: MatrixValue[];
  outputValues: MatrixValue[];
  accumulatorBefore: number;
  currentContribution: number;
  accumulatorAfter: number;
}

export interface MemoryTraffic {
  strategy: MemoryKernelStrategy;
  tileSize: number;
  outputElements: number;
  fmaCount: number;
  flops: number;
  globalReadScalars: number;
  globalWriteScalars: number;
  coldPathBytes: number;
  arithmeticIntensity: number;
  reusePerStagedScalar: number;
}

export interface MemoryResidency {
  hbmInputs: boolean;
  l2Tile: boolean;
  sharedTile: boolean;
  registerOperands: boolean;
  accumulator: boolean;
  hbmOutput: boolean;
}

export const MEMORY_X: number[][] = [
  [1, 2, 3, 4],
  [2, 1, 0, -1],
  [0, 1, 2, 1],
  [1, 0, -1, 2],
];

export const MEMORY_DY: number[][] = [
  [1, 0, 2, 1],
  [0, 1, 1, -1],
  [2, 1, 0, 1],
  [-1, 2, 1, 0],
];

const modifiableSteps: Record<MemoryStepId, MemoryStep> = {
  "global-request": {
    id: "global-request",
    compactLabel: "Global load",
    label: "Warp 发出 global memory load",
    activeLayer: "hbm",
    explanation: "指令携带的是 global address。若片上缓存没有数据，请求才继续到 HBM。",
  },
  "l2-lookup": {
    id: "l2-lookup",
    compactLabel: "L2",
    label: "请求经过 GPU 共享 L2 Cache",
    activeLayer: "l2",
    explanation: "L2 由硬件自动管理。命中可避免访问 HBM，未命中则把 cache line 从 HBM 带回。",
  },
  "shared-stage": {
    id: "shared-stage",
    compactLabel: "Shared",
    label: "Block 协作把输入 tile 放入 Shared Memory",
    activeLayer: "shared",
    explanation: "Shared Memory 是显式寻址的片上 scratchpad。一次装载后，Block 内多个线程可以重复读取。",
  },
  "register-load": {
    id: "register-load",
    compactLabel: "Registers",
    label: "线程把本轮两个操作数放进寄存器",
    activeLayer: "register",
    explanation: "寄存器属于线程。选中的线程只取计算 dW[m,n] 所需的 X[k,m] 与 dY[k,n]。",
  },
  fma: {
    id: "fma",
    compactLabel: "FMA",
    label: "执行乘加并更新寄存器 accumulator",
    activeLayer: "compute",
    explanation: "数据靠近计算单元后，线程连续执行乘加。累加器在所有 K slice 完成前不必写回 HBM。",
  },
  writeback: {
    id: "writeback",
    compactLabel: "Write back",
    label: "完整 dW tile 写回 global memory",
    activeLayer: "hbm",
    explanation: "完成全部 K slice 后才把结果写回。后续 kernel 通过同一 global address 读取这份梯度。",
  },
};

function assertMatrixCoordinate(name: string, value: number): void {
  if (!Number.isInteger(value) || value < 0 || value >= MEMORY_MATRIX_SIZE) {
    throw new RangeError(`${name} must be an integer from 0 to ${MEMORY_MATRIX_SIZE - 1}`);
  }
}

function assertTileSize(tileSize: number): void {
  if (![1, 2, 4].includes(tileSize)) {
    throw new RangeError("tileSize must be 1, 2, or 4");
  }
}

function rounded(value: number): number {
  return Number(value.toFixed(4));
}

export function multiplyWeightGradient(
  x: number[][] = MEMORY_X,
  dy: number[][] = MEMORY_DY,
): number[][] {
  const size = x.length;
  if (size === 0 || x.some((row) => row.length !== size)) {
    throw new Error("X must be a non-empty square matrix");
  }
  if (dy.length !== size || dy.some((row) => row.length !== size)) {
    throw new Error("dY must have the same square shape as X");
  }

  return Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, column) =>
      x.reduce((sum, xRow, k) => sum + xRow[row] * dy[k][column], 0),
    ),
  );
}

export const MEMORY_DW = multiplyWeightGradient();

export function gradientEquation(row: number, column: number): GradientEquation {
  assertMatrixCoordinate("row", row);
  assertMatrixCoordinate("column", column);
  const terms = Array.from({ length: MEMORY_MATRIX_SIZE }, (_, k) => ({
    k,
    left: MEMORY_X[k][row],
    right: MEMORY_DY[k][column],
    product: rounded(MEMORY_X[k][row] * MEMORY_DY[k][column]),
  }));
  return {
    row,
    column,
    terms,
    result: terms.reduce((sum, term) => sum + term.product, 0),
  };
}

export function effectiveMemoryTileSize(
  strategy: MemoryKernelStrategy,
  tileSize: number,
): number {
  assertTileSize(tileSize);
  return strategy === "naive" ? 1 : tileSize;
}

export function gradientTile(
  row: number,
  column: number,
  kStart: number,
  tileSize: number,
): GradientTile {
  assertMatrixCoordinate("row", row);
  assertMatrixCoordinate("column", column);
  assertTileSize(tileSize);
  if (!Number.isInteger(kStart) || kStart < 0 || kStart >= MEMORY_MATRIX_SIZE || kStart % tileSize !== 0) {
    throw new RangeError("kStart must identify a valid aligned K tile");
  }

  const outputRowStart = Math.floor(row / tileSize) * tileSize;
  const outputColumnStart = Math.floor(column / tileSize) * tileSize;
  const outputRows = Array.from({ length: tileSize }, (_, offset) => outputRowStart + offset);
  const outputColumns = Array.from({ length: tileSize }, (_, offset) => outputColumnStart + offset);
  const kValues = Array.from({ length: tileSize }, (_, offset) => kStart + offset);
  const xValues = kValues.flatMap((k) =>
    outputRows.map((matrixColumn) => ({
      id: `X[${k},${matrixColumn}]`,
      matrix: "X" as const,
      row: k,
      column: matrixColumn,
      value: MEMORY_X[k][matrixColumn],
      selectedOperand: matrixColumn === row,
    })),
  );
  const dyValues = kValues.flatMap((k) =>
    outputColumns.map((matrixColumn) => ({
      id: `dY[${k},${matrixColumn}]`,
      matrix: "dY" as const,
      row: k,
      column: matrixColumn,
      value: MEMORY_DY[k][matrixColumn],
      selectedOperand: matrixColumn === column,
    })),
  );
  const outputValues = outputRows.flatMap((outputRow) =>
    outputColumns.map((outputColumn) => ({
      id: `dW[${outputRow},${outputColumn}]`,
      matrix: "dW" as const,
      row: outputRow,
      column: outputColumn,
      value: MEMORY_DW[outputRow][outputColumn],
      selectedOperand: outputRow === row && outputColumn === column,
    })),
  );
  const accumulatorBefore = Array.from({ length: kStart }, (_, k) =>
    MEMORY_X[k][row] * MEMORY_DY[k][column]).reduce((sum, product) => sum + product, 0);
  const currentContribution = kValues.reduce(
    (sum, k) => sum + MEMORY_X[k][row] * MEMORY_DY[k][column],
    0,
  );

  return {
    tileSize,
    outputRowStart,
    outputColumnStart,
    kStart,
    outputRows,
    outputColumns,
    kValues,
    xValues,
    dyValues,
    outputValues,
    accumulatorBefore,
    currentContribution,
    accumulatorAfter: accumulatorBefore + currentContribution,
  };
}

export function memorySteps(strategy: MemoryKernelStrategy): MemoryStep[] {
  const ids: MemoryStepId[] = strategy === "tiled"
    ? ["global-request", "l2-lookup", "shared-stage", "register-load", "fma", "writeback"]
    : ["global-request", "l2-lookup", "register-load", "fma", "writeback"];
  return ids.map((id) => ({ ...modifiableSteps[id] }));
}

export function memoryResidency(
  strategy: MemoryKernelStrategy,
  stepId: MemoryStepId,
): MemoryResidency {
  const steps = memorySteps(strategy);
  const stepIndex = steps.findIndex((step) => step.id === stepId);
  if (stepIndex < 0) throw new Error(`${stepId} is unavailable for ${strategy}`);
  const reached = (id: MemoryStepId) => {
    const index = steps.findIndex((step) => step.id === id);
    return index >= 0 && stepIndex >= index;
  };

  return {
    hbmInputs: true,
    l2Tile: reached("l2-lookup"),
    sharedTile: strategy === "tiled" && reached("shared-stage"),
    registerOperands: reached("register-load"),
    accumulator: reached("fma"),
    hbmOutput: reached("writeback"),
  };
}

export function memoryTraffic(
  strategy: MemoryKernelStrategy,
  tileSize: number,
): MemoryTraffic {
  assertTileSize(tileSize);
  const effectiveTileSize = effectiveMemoryTileSize(strategy, tileSize);
  const size = MEMORY_MATRIX_SIZE;
  const outputElements = size * size;
  const fmaCount = outputElements * size;
  const globalReadScalars = strategy === "naive"
    ? 2 * outputElements * size
    : 2 * outputElements * size / effectiveTileSize;
  const globalWriteScalars = outputElements;
  const coldPathBytes = (globalReadScalars + globalWriteScalars) * MEMORY_FLOAT_BYTES;
  const flops = fmaCount * 2;

  return {
    strategy,
    tileSize: effectiveTileSize,
    outputElements,
    fmaCount,
    flops,
    globalReadScalars,
    globalWriteScalars,
    coldPathBytes,
    arithmeticIntensity: rounded(flops / coldPathBytes),
    reusePerStagedScalar: strategy === "tiled" ? effectiveTileSize : 1,
  };
}
