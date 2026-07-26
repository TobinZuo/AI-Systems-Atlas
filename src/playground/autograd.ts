export type AutogradMode = "tracked" | "no-grad" | "in-place";
export type AutogradDevice = "cpu" | "cuda";

export type AutogradPhase =
  | "setup"
  | "forward"
  | "record"
  | "mutate"
  | "seed"
  | "backward"
  | "accumulate"
  | "release"
  | "error";

export type AutogradNodeId =
  | "w"
  | "x"
  | "b"
  | "target"
  | "wx"
  | "regularizer"
  | "prediction"
  | "residual"
  | "loss";

export interface AutogradConfig {
  device: AutogradDevice;
  w: number;
  x: number;
  b: number;
  target: number;
  includeRegularizer: boolean;
  mode: AutogradMode;
  backwardPasses: 1 | 2;
  zeroBetweenPasses: boolean;
}

export interface GradientContribution {
  source: AutogradNodeId;
  value: number;
  equation: string;
}

export interface AutogradNode {
  id: AutogradNodeId;
  label: string;
  expression: string;
  value: number;
  kind: "leaf" | "operation" | "output";
  requiresGrad: boolean;
  isLeaf: boolean;
  gradFn: string | null;
  saved: string[];
  version: number | null;
  backwardGradient: number | null;
  storedGrad: number | null;
  contributions: GradientContribution[];
}

export interface AutogradEdge {
  from: AutogradNodeId;
  to: AutogradNodeId;
  forwardLabel: string;
  localDerivative: number | null;
  gradientContribution: number | null;
}

export interface AutogradStep {
  id: string;
  phase: AutogradPhase;
  compactLabel: string;
  label: string;
  call: string;
  explanation: string;
  activeNodeIds: AutogradNodeId[];
  reads: string[];
  writes: string[];
  status: "executed" | "fault" | "blocked";
}

export interface AutogradSimulation {
  config: AutogradConfig;
  nodes: AutogradNode[];
  edges: AutogradEdge[];
  steps: AutogradStep[];
  graphRecorded: boolean;
  backwardSucceeded: boolean;
  errorCode: "requires_grad=false" | "saved tensor version mismatch" | null;
  forward: {
    wx: number;
    regularizer: number;
    prediction: number;
    residual: number;
    loss: number;
  };
  onePassGradients: { w: number; b: number };
  finalGradients: { w: number; b: number } | null;
  savedTensorCount: number;
  graphGenerationCount: number;
  gradientMultiplier: number;
}

function formatNumber(value: number): string {
  if (Object.is(value, -0)) return "0";
  if (Number.isInteger(value)) return String(value);
  return String(Number(value.toFixed(4)));
}

function validateConfig(config: AutogradConfig): void {
  if (config.device !== "cpu" && config.device !== "cuda") {
    throw new Error("device must be cpu or cuda");
  }
  for (const [name, value] of Object.entries({
    w: config.w,
    x: config.x,
    b: config.b,
    target: config.target,
  })) {
    if (!Number.isFinite(value)) throw new Error(`${name} must be finite`);
  }

  if (config.backwardPasses !== 1 && config.backwardPasses !== 2) {
    throw new Error("backwardPasses must be 1 or 2");
  }
  if (config.mode === "in-place" && !config.includeRegularizer) {
    throw new Error("in-place mode needs the regularizer branch that saves w");
  }
  if (config.mode !== "tracked" && config.backwardPasses !== 1) {
    throw new Error("fault demonstrations support one backward request");
  }
}

function makeStep(
  id: string,
  phase: AutogradPhase,
  compactLabel: string,
  label: string,
  call: string,
  explanation: string,
  activeNodeIds: AutogradNodeId[],
  reads: string[],
  writes: string[],
  status: AutogradStep["status"] = "executed",
): AutogradStep {
  return {
    id,
    phase,
    compactLabel,
    label,
    call,
    explanation,
    activeNodeIds,
    reads,
    writes,
    status,
  };
}

export function simulateAutograd(config: AutogradConfig): AutogradSimulation {
  validateConfig(config);

  const tracked = config.mode !== "no-grad";
  const wx = config.w * config.x;
  const regularizer = config.includeRegularizer ? 0.5 * config.w * config.w : 0;
  const prediction = wx + regularizer + config.b;
  const residual = prediction - config.target;
  const loss = 0.5 * residual * residual;
  const gradWFromWx = residual * config.x;
  const gradWFromRegularizer = config.includeRegularizer ? residual * config.w : 0;
  const onePassGradients = {
    w: gradWFromWx + gradWFromRegularizer,
    b: residual,
  };
  const backwardSucceeded = config.mode === "tracked";
  const gradientMultiplier = config.backwardPasses === 2 && !config.zeroBetweenPasses ? 2 : 1;
  const finalGradients = backwardSucceeded
    ? {
        w: onePassGradients.w * gradientMultiplier,
        b: onePassGradients.b * gradientMultiplier,
      }
    : null;

  const operationGrad = tracked ? residual : null;
  const wContributions: GradientContribution[] = backwardSucceeded
    ? [
        {
          source: "wx",
          value: gradWFromWx,
          equation: `${formatNumber(residual)} × x(${formatNumber(config.x)})`,
        },
        ...(config.includeRegularizer
          ? [{
              source: "regularizer" as const,
              value: gradWFromRegularizer,
              equation: `${formatNumber(residual)} × w(${formatNumber(config.w)})`,
            }]
          : []),
      ]
    : [];

  const nodes: AutogradNode[] = [
    {
      id: "w",
      label: "w",
      expression: config.mode === "in-place"
        ? `initial ${formatNumber(config.w)}; current ${config.device} storage ${formatNumber(config.w + 1)}`
        : `torch.tensor(${formatNumber(config.w)}, device="${config.device}", requires_grad=True)`,
      value: config.mode === "in-place" ? config.w + 1 : config.w,
      kind: "leaf",
      requiresGrad: true,
      isLeaf: true,
      gradFn: null,
      saved: [],
      version: config.mode === "in-place" ? 1 : 0,
      backwardGradient: backwardSucceeded ? onePassGradients.w : null,
      storedGrad: finalGradients?.w ?? null,
      contributions: wContributions,
    },
    {
      id: "x",
      label: "x",
      expression: `torch.tensor(${formatNumber(config.x)}, device="${config.device}")`,
      value: config.x,
      kind: "leaf",
      requiresGrad: false,
      isLeaf: true,
      gradFn: null,
      saved: [],
      version: 0,
      backwardGradient: null,
      storedGrad: null,
      contributions: [],
    },
    {
      id: "b",
      label: "b",
      expression: `torch.tensor(${formatNumber(config.b)}, device="${config.device}", requires_grad=True)`,
      value: config.b,
      kind: "leaf",
      requiresGrad: true,
      isLeaf: true,
      gradFn: null,
      saved: [],
      version: 0,
      backwardGradient: backwardSucceeded ? residual : null,
      storedGrad: finalGradients?.b ?? null,
      contributions: backwardSucceeded
        ? [{ source: "prediction", value: residual, equation: `${formatNumber(residual)} × 1` }]
        : [],
    },
    {
      id: "target",
      label: "target",
      expression: `torch.tensor(${formatNumber(config.target)}, device="${config.device}")`,
      value: config.target,
      kind: "leaf",
      requiresGrad: false,
      isLeaf: true,
      gradFn: null,
      saved: [],
      version: 0,
      backwardGradient: null,
      storedGrad: null,
      contributions: [],
    },
    {
      id: "wx",
      label: "u = w × x",
      expression: `${formatNumber(config.w)} × ${formatNumber(config.x)}`,
      value: wx,
      kind: "operation",
      requiresGrad: tracked,
      isLeaf: false,
      gradFn: tracked ? "MulBackward0" : null,
      saved: tracked ? [`x = ${formatNumber(config.x)}`] : [],
      version: null,
      backwardGradient: operationGrad,
      storedGrad: null,
      contributions: [],
    },
    ...(config.includeRegularizer
      ? [{
          id: "regularizer" as const,
          label: "r = 0.5 × w²",
          expression: `0.5 × ${formatNumber(config.w)}²`,
          value: regularizer,
          kind: "operation" as const,
          requiresGrad: tracked,
          isLeaf: false,
          gradFn: tracked ? "PowBackward0 + MulBackward0" : null,
          saved: tracked ? [`saved w reference; forward value was ${formatNumber(config.w)}`, "saved version = 0"] : [],
          version: null,
          backwardGradient: operationGrad,
          storedGrad: null,
          contributions: [],
        }]
      : []),
    {
      id: "prediction",
      label: "prediction",
      expression: config.includeRegularizer ? "u + r + b" : "u + b",
      value: prediction,
      kind: "operation",
      requiresGrad: tracked,
      isLeaf: false,
      gradFn: tracked ? "AddBackward0" : null,
      saved: [],
      version: null,
      backwardGradient: operationGrad,
      storedGrad: null,
      contributions: [],
    },
    {
      id: "residual",
      label: "residual",
      expression: "prediction - target",
      value: residual,
      kind: "operation",
      requiresGrad: tracked,
      isLeaf: false,
      gradFn: tracked ? "SubBackward0" : null,
      saved: [],
      version: null,
      backwardGradient: tracked ? residual : null,
      storedGrad: null,
      contributions: [],
    },
    {
      id: "loss",
      label: "loss",
      expression: "0.5 × residual²",
      value: loss,
      kind: "output",
      requiresGrad: tracked,
      isLeaf: false,
      gradFn: tracked ? "PowBackward0 + MulBackward0" : null,
      saved: tracked ? [`residual = ${formatNumber(residual)}`] : [],
      version: null,
      backwardGradient: tracked ? 1 : null,
      storedGrad: null,
      contributions: [],
    },
  ];

  const edges: AutogradEdge[] = [
    { from: "w", to: "wx", forwardLabel: "w", localDerivative: tracked ? config.x : null, gradientContribution: tracked ? gradWFromWx : null },
    { from: "x", to: "wx", forwardLabel: "x", localDerivative: null, gradientContribution: null },
    ...(config.includeRegularizer
      ? [{ from: "w" as const, to: "regularizer" as const, forwardLabel: "w", localDerivative: tracked ? config.w : null, gradientContribution: tracked ? gradWFromRegularizer : null }]
      : []),
    { from: "wx", to: "prediction", forwardLabel: "u", localDerivative: tracked ? 1 : null, gradientContribution: tracked ? residual : null },
    ...(config.includeRegularizer
      ? [{ from: "regularizer" as const, to: "prediction" as const, forwardLabel: "r", localDerivative: tracked ? 1 : null, gradientContribution: tracked ? residual : null }]
      : []),
    { from: "b", to: "prediction", forwardLabel: "b", localDerivative: tracked ? 1 : null, gradientContribution: tracked ? residual : null },
    { from: "prediction", to: "residual", forwardLabel: "prediction", localDerivative: tracked ? 1 : null, gradientContribution: tracked ? residual : null },
    { from: "target", to: "residual", forwardLabel: "target", localDerivative: null, gradientContribution: null },
    { from: "residual", to: "loss", forwardLabel: "residual", localDerivative: tracked ? residual : null, gradientContribution: tracked ? residual : null },
  ];

  const setupStep = makeStep(
    "create-leaves",
    "setup",
    "创建叶子",
    "Tensor 保存数据和 AutogradMeta",
    "w.requires_grad_(True); b.requires_grad_(True)",
    "w 和 b 是需要优化的叶子 Tensor。它们没有 grad_fn，因为不是某个已记录算子的输出；最终梯度会写进它们的 .grad。",
    ["w", "x", "b", "target"],
    ["Python scalar values"],
    ["Tensor storage", "requires_grad", "version counter = 0"],
  );

  const forwardStep = makeStep(
    "forward-values",
    "forward",
    "执行 Forward",
    "算子先计算真实数值",
    config.includeRegularizer
      ? "u = w * x; r = 0.5 * w.pow(2); prediction = u + r + b"
      : "u = w * x; prediction = u + b",
    tracked
      ? "每个 Tensor 算子一边执行前向计算，一边让 Autograd 为需要求导的输出连接 backward Node。"
      : "no_grad 只关闭记录，不关闭数值计算。prediction 和 loss 仍然算出来，但不会留下可反向遍历的 Node。",
    config.includeRegularizer
      ? ["w", "x", "b", "wx", "regularizer", "prediction"]
      : ["w", "x", "b", "wx", "prediction"],
    ["leaf Tensor values"],
    ["intermediate Tensor values", tracked ? "backward edges" : "values only"],
  );

  const lossStep = makeStep(
    "build-loss",
    "record",
    tracked ? "连接计算图" : "不记录图",
    tracked ? "输出通过 grad_fn 指回反向图" : "输出的 grad_fn 是 None",
    "residual = prediction - target; loss = 0.5 * residual.pow(2)",
    tracked
      ? "loss.grad_fn 是进入反向图的入口。图记录的是已执行的算子关系，不是保存一份 Python 源代码。"
      : "即使输入 w.requires_grad=True，no_grad 里的输出也表现为 requires_grad=False，因此 loss.backward() 没有入口。",
    ["prediction", "target", "residual", "loss"],
    ["prediction", "target"],
    tracked ? ["loss.grad_fn", "next_edges"] : ["loss.requires_grad = False", "loss.grad_fn = None"],
  );

  let steps: AutogradStep[];
  if (config.mode === "no-grad") {
    steps = [
      setupStep,
      makeStep(
        "enter-no-grad",
        "record",
        "关闭记录",
        "线程局部的 grad mode 被临时关闭",
        "with torch.no_grad():",
        "Autograd 不为上下文中的运算创建 backward Node。退出后默认 grad mode 可以恢复。",
        [],
        ["thread-local grad mode"],
        ["is_grad_enabled = False"],
      ),
      forwardStep,
      lossStep,
      makeStep(
        "backward-request",
        "seed",
        "请求 Backward",
        "loss 没有反向图入口",
        "loss.backward()",
        "Backward 需要从一个 requires_grad=True 且拥有 grad_fn 的输出开始；当前 loss 不满足这个契约。",
        ["loss"],
        ["loss.requires_grad", "loss.grad_fn"],
        [],
        "fault",
      ),
      makeStep(
        "no-grad-error",
        "error",
        "停止训练步",
        "没有梯度可以写回叶子 Tensor",
        "RuntimeError: loss does not require grad",
        "no_grad 不是把图建好后再删除，而是在 forward 时就不记录。本次 backward 必须失败，w.grad 和 b.grad 仍为空。",
        ["w", "b", "loss"],
        ["failed backward request"],
        ["error surfaced to Python"],
        "fault",
      ),
    ];
  } else if (config.mode === "in-place") {
    steps = [
      setupStep,
      forwardStep,
      lossStep,
      makeStep(
        "save-for-backward",
        "record",
        "保存反向所需值",
        "PowBackward 保存 w 及其版本号",
        "ctx.save_for_backward(w)  # conceptual",
        "为了在 backward 计算 d(0.5×w²)/dw，节点需要 forward 时的 w。它同时记住 w 的 version counter 为 0。",
        ["w", "regularizer"],
        ["w storage", "w._version = 0"],
        ["saved reference", "saved version = 0"],
      ),
      makeStep(
        "mutate-parameter",
        "mutate",
        "原地修改 w",
        "同一份 storage 被改写，版本号递增",
        "with torch.no_grad(): w.add_(1)",
        `no_grad 允许 optimizer 风格的原地更新不进入新计算图，但正常更新应发生在 backward 之后。这里故意提前把值从 ${formatNumber(config.w)} 改成 ${formatNumber(config.w + 1)}，w._version 也从 0 增加到 1。`,
        ["w"],
        ["w storage", "version = 0"],
        ["mutated storage", "version = 1"],
      ),
      makeStep(
        "seed-backward",
        "seed",
        "种下梯度 1",
        "标量 loss 从 dL/dL = 1 开始",
        "loss.backward()",
        "Autograd Engine 从 loss.grad_fn 出发，把标量输出的初始上游梯度设为 1。",
        ["loss"],
        ["loss.grad_fn"],
        ["ready queue", "dL/dL = 1"],
      ),
      makeStep(
        "version-check-fault",
        "error",
        "版本检查失败",
        "Backward 拒绝读取已经改变的 saved tensor",
        "saved_version(0) != current_version(1)",
        "如果继续使用被改写的 w，梯度会对应错误的 forward。版本检查主动报错，避免静默产生错误梯度。",
        ["w", "regularizer"],
        ["saved version = 0", "current version = 1"],
        ["RuntimeError", "backward aborted"],
        "fault",
      ),
      makeStep(
        "blocked-accumulation",
        "error",
        "不写入 .grad",
        "失败的 backward 没有有效参数梯度",
        "w.grad is None; b.grad is None",
        "训练框架应停止当前 step。不能把部分计算结果交给 optimizer。",
        ["w", "b"],
        ["aborted graph"],
        ["no optimizer update"],
        "blocked",
      ),
    ];
  } else {
    steps = [
      setupStep,
      forwardStep,
      lossStep,
      makeStep(
        "save-for-backward",
        "record",
        "保存必要值",
        "Backward Node 只保留局部求导需要的数据",
        "save x, w, residual for backward",
        "Autograd 不需要复制所有 forward Tensor。每个 Node 只保存自己的 backward 公式真正需要的值和版本。",
        config.includeRegularizer ? ["wx", "regularizer", "loss"] : ["wx", "loss"],
        ["x", ...(config.includeRegularizer ? ["w"] : []), "residual"],
        ["saved tensors", "saved versions"],
      ),
      makeStep(
        "seed-backward",
        "seed",
        "种下梯度 1",
        "标量 loss 从 dL/dL = 1 开始",
        "loss.backward()  # implicit gradient = 1",
        "标量输出不需要显式传入上游梯度。Autograd Engine 把 loss 的梯度设为 1，并把根节点放入 ready queue。",
        ["loss"],
        ["loss.grad_fn"],
        ["ready queue", "dL/dL = 1"],
      ),
      makeStep(
        "reverse-chain",
        "backward",
        "反向穿过 Loss",
        "每个 Node 计算 vector-Jacobian product",
        `dL/dprediction = 1 × residual(${formatNumber(residual)})`,
        "Engine 只有在一个 Node 的所有下游梯度都到齐后才调度它。这里从 loss 依次回到 residual 和 prediction。",
        ["loss", "residual", "prediction"],
        ["upstream gradient", "saved residual"],
        [`dL/dprediction = ${formatNumber(residual)}`],
      ),
      makeStep(
        "branch-backward",
        "backward",
        "沿分支回传",
        config.includeRegularizer ? "w 同时收到两条梯度贡献" : "w 从乘法分支收到梯度",
        config.includeRegularizer
          ? `via u: ${formatNumber(residual)}×${formatNumber(config.x)}; via r: ${formatNumber(residual)}×${formatNumber(config.w)}`
          : `via u: ${formatNumber(residual)}×${formatNumber(config.x)}`,
        config.includeRegularizer
          ? "链式法则先在每条边计算局部贡献。因为同一个叶子 w 被两条路径使用，贡献必须相加，不能互相覆盖。"
          : "乘法 backward 使用 forward 保存的 x，把上游梯度乘以局部导数 x。",
        config.includeRegularizer ? ["wx", "regularizer", "w"] : ["wx", "w"],
        ["upstream gradient", "saved forward values"],
        wContributions.map((item) => `${item.source}: ${formatNumber(item.value)}`),
      ),
      makeStep(
        "accumulate-leaves",
        "accumulate",
        "写入叶子 .grad",
        "AccumulateGrad 把所有贡献加到参数梯度",
        config.includeRegularizer
          ? `w.grad += ${formatNumber(gradWFromWx)} + ${formatNumber(gradWFromRegularizer)}`
          : `w.grad += ${formatNumber(gradWFromWx)}`,
        config.backwardPasses === 2
          ? config.zeroBetweenPasses
            ? "每次新 forward 都重建一张图；第二次 backward 前清空 .grad，所以最终只保留第二次的梯度。"
            : "每次新 forward 都重建一张图；因为没有清空 .grad，第二次 backward 会继续相加，最终梯度变为单次的两倍。"
          : "中间 Tensor 默认不保留 .grad。叶子参数的 AccumulateGrad Node 将贡献写入 w.grad 和 b.grad。",
        ["w", "b"],
        wContributions.map((item) => `${item.source} contribution`),
        [`w.grad = ${formatNumber(finalGradients!.w)}`, `b.grad = ${formatNumber(finalGradients!.b)}`],
      ),
      makeStep(
        "release-graph",
        "release",
        "释放反向图",
        "本轮 saved tensors 可以回收，参数和 .grad 继续存在",
        "backward(retain_graph=False)",
        "默认 backward 完成后会释放这张动态图持有的中间状态。下一轮 forward 按实际执行路径重新构图。",
        ["w", "b"],
        ["completed backward graph"],
        ["leaf .grad remains", "saved tensors released"],
      ),
    ];
  }

  return {
    config: { ...config },
    nodes,
    edges,
    steps,
    graphRecorded: tracked,
    backwardSucceeded,
    errorCode: config.mode === "no-grad"
      ? "requires_grad=false"
      : config.mode === "in-place"
        ? "saved tensor version mismatch"
        : null,
    forward: { wx, regularizer, prediction, residual, loss },
    onePassGradients,
    finalGradients,
    savedTensorCount: tracked ? (config.includeRegularizer ? 3 : 2) : 0,
    graphGenerationCount: config.mode === "tracked" ? config.backwardPasses : 1,
    gradientMultiplier,
  };
}

export function getAutogradNode(
  simulation: AutogradSimulation,
  nodeId: AutogradNodeId,
): AutogradNode | undefined {
  return simulation.nodes.find((node) => node.id === nodeId);
}
