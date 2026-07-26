import { ArrowCounterClockwise } from "@phosphor-icons/react/ArrowCounterClockwise";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { BracketsCurly } from "@phosphor-icons/react/BracketsCurly";
import { CaretLeft } from "@phosphor-icons/react/CaretLeft";
import { CaretRight } from "@phosphor-icons/react/CaretRight";
import { CheckCircle } from "@phosphor-icons/react/CheckCircle";
import { Code } from "@phosphor-icons/react/Code";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Database } from "@phosphor-icons/react/Database";
import { FlowArrow } from "@phosphor-icons/react/FlowArrow";
import { Function as FunctionIcon } from "@phosphor-icons/react/Function";
import { Graph } from "@phosphor-icons/react/Graph";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { Pause } from "@phosphor-icons/react/Pause";
import { Play } from "@phosphor-icons/react/Play";
import { Warning } from "@phosphor-icons/react/Warning";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  getAutogradNode,
  simulateAutograd,
  type AutogradDevice,
  type AutogradMode,
  type AutogradNode,
  type AutogradNodeId,
  type AutogradPhase,
} from "../playground/autograd";

const numberOptions = {
  w: [1, 2, 3],
  x: [2, 3, 4],
  b: [0, 1, 2],
  target: [8, 10, 12],
};

const phaseLabels: Record<AutogradPhase, string> = {
  setup: "Tensor setup",
  forward: "Forward compute",
  record: "Graph recording",
  mutate: "Storage mutation",
  seed: "Backward seed",
  backward: "Reverse traversal",
  accumulate: "Leaf accumulation",
  release: "Graph lifetime",
  error: "Safety boundary",
};

function formatNumber(value: number | null): string {
  if (value === null) return "None";
  if (Object.is(value, -0)) return "0";
  return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(4)));
}

function PhaseIcon({ phase }: { phase: AutogradPhase }) {
  const icons: Record<AutogradPhase, ReactNode> = {
    setup: <Database size={18} weight="duotone" />,
    forward: <Lightning size={18} weight="duotone" />,
    record: <Graph size={18} weight="duotone" />,
    mutate: <BracketsCurly size={18} weight="duotone" />,
    seed: <Play size={18} weight="duotone" />,
    backward: <FlowArrow size={18} weight="duotone" />,
    accumulate: <FunctionIcon size={18} weight="duotone" />,
    release: <CheckCircle size={18} weight="duotone" />,
    error: <Warning size={18} weight="duotone" />,
  };
  return icons[phase];
}

function NodeButton({
  node,
  selected,
  active,
  fault,
  displayValue,
  onSelect,
}: {
  node: AutogradNode;
  selected: boolean;
  active: boolean;
  fault: boolean;
  displayValue?: number;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      className={`autograd-node kind-${node.kind}${selected ? " is-selected" : ""}${active ? " is-active" : ""}${!node.gradFn && !node.isLeaf ? " is-untracked" : ""}${fault ? " is-fault" : ""}`}
      aria-pressed={selected}
      onClick={onSelect}
    >
      <span>{node.isLeaf ? "Leaf Tensor" : node.kind === "output" ? "Scalar output" : node.gradFn ?? "Value only"}</span>
      <strong>{node.label}</strong>
      <code>{formatNumber(displayValue ?? node.value)}</code>
      <small>{node.requiresGrad ? "requires_grad=True" : "requires_grad=False"}</small>
    </button>
  );
}

export function AutogradPlayground() {
  const [device, setDevice] = useState<AutogradDevice>("cuda");
  const [w, setW] = useState(2);
  const [x, setX] = useState(3);
  const [b, setB] = useState(1);
  const [target, setTarget] = useState(10);
  const [includeRegularizer, setIncludeRegularizer] = useState(true);
  const [mode, setMode] = useState<AutogradMode>("tracked");
  const [backwardPasses, setBackwardPasses] = useState<1 | 2>(1);
  const [zeroBetweenPasses, setZeroBetweenPasses] = useState(true);
  const [stepIndex, setStepIndex] = useState(0);
  const [selectedNodeId, setSelectedNodeId] = useState<AutogradNodeId>("loss");
  const [isPlaying, setIsPlaying] = useState(false);

  const simulation = useMemo(
    () => simulateAutograd({
      device,
      w,
      x,
      b,
      target,
      includeRegularizer,
      mode,
      backwardPasses,
      zeroBetweenPasses,
    }),
    [device, w, x, b, target, includeRegularizer, mode, backwardPasses, zeroBetweenPasses],
  );
  const visibleStepIndex = Math.min(stepIndex, simulation.steps.length - 1);
  const currentStep = simulation.steps[visibleStepIndex];
  const selectedNode = getAutogradNode(simulation, selectedNodeId)
    ?? simulation.nodes[simulation.nodes.length - 1];
  const activeNodeIds = new Set(currentStep.activeNodeIds);
  const isBackwardPhase = ["seed", "backward", "accumulate", "release", "error"].includes(currentStep.phase);
  const mutationStepIndex = simulation.steps.findIndex((step) => step.id === "mutate-parameter");
  const versionFaultIndex = simulation.steps.findIndex((step) => step.id === "version-check-fault");
  const saveStepIndex = simulation.steps.findIndex((step) => step.id === "save-for-backward");
  const contributionStepIndex = simulation.steps.findIndex((step) => step.id === "branch-backward");
  const accumulationStepIndex = simulation.steps.findIndex((step) => step.id === "accumulate-leaves");
  const releaseStepIndex = simulation.steps.findIndex((step) => step.id === "release-graph");
  const mutationReached = mode === "in-place" && mutationStepIndex !== -1 && visibleStepIndex >= mutationStepIndex;
  const versionFault = mode === "in-place" && versionFaultIndex !== -1 && visibleStepIndex >= versionFaultIndex;
  const savedValuesVisible = saveStepIndex !== -1
    && visibleStepIndex >= saveStepIndex
    && (releaseStepIndex === -1 || visibleStepIndex < releaseStepIndex);
  const backwardValuesVisible = simulation.backwardSucceeded
    && ["backward", "accumulate", "release"].includes(currentStep.phase);
  const contributionsVisible = simulation.backwardSucceeded
    && contributionStepIndex !== -1
    && visibleStepIndex >= contributionStepIndex;
  const storedGradVisible = simulation.backwardSucceeded
    && accumulationStepIndex !== -1
    && visibleStepIndex >= accumulationStepIndex;

  useEffect(() => {
    if (!isPlaying) return undefined;
    const timer = window.setInterval(() => {
      setStepIndex((current) => {
        if (current >= simulation.steps.length - 1) {
          setIsPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 1150);
    return () => window.clearInterval(timer);
  }, [isPlaying, simulation.steps.length]);

  useEffect(() => {
    if (stepIndex >= simulation.steps.length) setStepIndex(simulation.steps.length - 1);
    if (!simulation.nodes.some((node) => node.id === selectedNodeId)) setSelectedNodeId("loss");
  }, [simulation, selectedNodeId, stepIndex]);

  const restart = () => {
    setStepIndex(0);
    setIsPlaying(false);
  };

  const selectMode = (nextMode: AutogradMode) => {
    setMode(nextMode);
    setBackwardPasses(1);
    setZeroBetweenPasses(true);
    if (nextMode === "in-place") setIncludeRegularizer(true);
    restart();
  };

  const reset = () => {
    setDevice("cuda");
    setW(2);
    setX(3);
    setB(1);
    setTarget(10);
    setIncludeRegularizer(true);
    setMode("tracked");
    setBackwardPasses(1);
    setZeroBetweenPasses(true);
    setSelectedNodeId("loss");
    restart();
  };

  const leaf = (nodeId: AutogradNodeId) => getAutogradNode(simulation, nodeId)!;

  return (
    <section className="autograd-lab" id="autograd-lab" aria-labelledby="autograd-lab-title">
      <header className="autograd-lab-header">
        <div>
          <span>Dynamic graph lab</span>
          <h2 id="autograd-lab-title">Forward 一边算数值，一边留下 Backward 可以执行的路径</h2>
          <p>改变标量、移除分支或注入错误。图结构、saved tensor、链式法则和叶子梯度会一起变化。</p>
        </div>
        <pre aria-label="本页使用的标量计算"><code>{includeRegularizer
          ? "u = w*x; r = 0.5*w²; pred = u+r+b; loss = 0.5*(pred-y)²"
          : "u = w*x; pred = u+b; loss = 0.5*(pred-y)²"}</code></pre>
      </header>

      <div className="autograd-mode-bar" aria-label="Autograd 运行模式">
        <div role="tablist" aria-label="Grad mode">
          <button type="button" role="tab" aria-selected={mode === "tracked"} className={mode === "tracked" ? "is-active" : ""} onClick={() => selectMode("tracked")}><Graph size={16} weight="duotone" />正常记录</button>
          <button type="button" role="tab" aria-selected={mode === "no-grad"} className={mode === "no-grad" ? "is-active" : ""} onClick={() => selectMode("no-grad")}><Code size={16} weight="duotone" />torch.no_grad</button>
          <button type="button" role="tab" aria-selected={mode === "in-place"} className={mode === "in-place" ? "is-active is-error" : ""} onClick={() => selectMode("in-place")}><Warning size={16} weight="duotone" />原地修改</button>
        </div>
        <p>{mode === "tracked" ? "完整记录并执行 backward。" : mode === "no-grad" ? "Forward 有数值，但没有 backward graph。" : "Forward 后改写 saved tensor，触发版本检查。"}</p>
      </div>

      <div className="autograd-config-bar" aria-label="标量计算配置">
        {(["w", "x", "b", "target"] as const).map((name) => {
          const value = { w, x, b, target }[name];
          const setters = { w: setW, x: setX, b: setB, target: setTarget };
          return (
            <label key={name}>
              <span>{name}</span>
              <select value={value} onChange={(event) => { setters[name](Number(event.target.value)); restart(); }}>
                {numberOptions[name].map((option) => <option value={option} key={option}>{option}</option>)}
              </select>
            </label>
          );
        })}
        <label>
          <span>Tensor device</span>
          <select value={device} onChange={(event) => { setDevice(event.target.value as AutogradDevice); restart(); }}>
            <option value="cuda">CUDA</option>
            <option value="cpu">CPU</option>
          </select>
        </label>
        <label className="autograd-branch-control">
          <span>动态图分支</span>
          <button type="button" aria-pressed={includeRegularizer} disabled={mode === "in-place"} onClick={() => { setIncludeRegularizer((current) => !current); restart(); }}>
            {includeRegularizer ? "包含 0.5 × w²" : "仅包含 w × x"}
          </button>
        </label>
        <div className="autograd-play-actions">
          <button type="button" className="autograd-play-button" onClick={() => {
            if (!isPlaying && stepIndex === simulation.steps.length - 1) setStepIndex(0);
            setIsPlaying((current) => !current);
          }}>{isPlaying ? <Pause size={16} weight="fill" /> : <Play size={16} weight="fill" />}{isPlaying ? "暂停" : "播放"}</button>
          <button type="button" onClick={reset} aria-label="重置 Autograd 实验"><ArrowCounterClockwise size={17} /></button>
        </div>
      </div>

      <div className="autograd-summary" aria-label="Autograd 执行摘要">
        <div><span>Forward result</span><strong>loss = {formatNumber(simulation.forward.loss)}</strong><small>prediction = {formatNumber(simulation.forward.prediction)}</small></div>
        <div><span>Backward graph</span><strong>{simulation.graphRecorded ? `${simulation.nodes.filter((node) => node.gradFn).length} recorded Nodes` : "0 recorded Nodes"}</strong><small>{simulation.savedTensorCount} 个 saved values</small></div>
        <div><span>数学导数</span><strong>∂L/∂w = {formatNumber(simulation.onePassGradients.w)}</strong><small>有效 backward 单轮应得到 ∂L/∂b = {formatNumber(simulation.onePassGradients.b)}</small></div>
        <div className={simulation.backwardSucceeded ? "is-valid" : "is-error"}><span>叶子 .grad</span><strong>{simulation.finalGradients ? `w.grad = ${formatNumber(simulation.finalGradients.w)}` : simulation.errorCode}</strong><small>{simulation.finalGradients ? `b.grad = ${formatNumber(simulation.finalGradients.b)}` : "optimizer 不可执行"}</small></div>
      </div>

      <div className="autograd-step-rail" role="tablist" aria-label="Autograd 执行阶段">
        {simulation.steps.map((step, index) => (
          <button
            type="button"
            role="tab"
            aria-selected={index === stepIndex}
            className={`${index === stepIndex ? "is-active" : ""}${index < stepIndex ? " is-reached" : ""} status-${step.status}`}
            onClick={() => { setStepIndex(index); setIsPlaying(false); }}
            key={step.id}
          >
            <span>{index + 1}</span>
            <strong>{step.compactLabel}</strong>
          </button>
        ))}
      </div>

      <div className="autograd-main-grid">
        <section className={`autograd-graph-panel phase-${currentStep.phase}`} aria-labelledby="autograd-graph-title">
          <header>
            <div><span>{isBackwardPhase ? "Backward view" : "Forward view"}</span><h3 id="autograd-graph-title">实际执行路径就是本轮计算图</h3></div>
            <code>{includeRegularizer ? "w has 2 paths" : "w has 1 path"}</code>
          </header>

          <div className={`autograd-graph-flow${isBackwardPhase ? " is-backward" : ""}`}>
            <div className="autograd-node-column leaf-column">
              <NodeButton node={leaf("w")} selected={selectedNode.id === "w"} active={activeNodeIds.has("w")} fault={versionFault} displayValue={mode === "in-place" && !mutationReached ? w : undefined} onSelect={() => setSelectedNodeId("w")} />
              <NodeButton node={leaf("x")} selected={selectedNode.id === "x"} active={activeNodeIds.has("x")} fault={false} onSelect={() => setSelectedNodeId("x")} />
              <NodeButton node={leaf("b")} selected={selectedNode.id === "b"} active={activeNodeIds.has("b")} fault={false} onSelect={() => setSelectedNodeId("b")} />
              <NodeButton node={leaf("target")} selected={selectedNode.id === "target"} active={activeNodeIds.has("target")} fault={false} onSelect={() => setSelectedNodeId("target")} />
            </div>
            <ArrowRight className="autograd-stage-arrow" size={19} aria-hidden="true" />
            <div className="autograd-node-column branch-column">
              <NodeButton node={leaf("wx")} selected={selectedNode.id === "wx"} active={activeNodeIds.has("wx")} fault={false} onSelect={() => setSelectedNodeId("wx")} />
              {includeRegularizer && <NodeButton node={leaf("regularizer")} selected={selectedNode.id === "regularizer"} active={activeNodeIds.has("regularizer")} fault={versionFault} onSelect={() => setSelectedNodeId("regularizer")} />}
            </div>
            <ArrowRight className="autograd-stage-arrow" size={19} aria-hidden="true" />
            <div className="autograd-node-column single-column">
              <NodeButton node={leaf("prediction")} selected={selectedNode.id === "prediction"} active={activeNodeIds.has("prediction")} fault={false} onSelect={() => setSelectedNodeId("prediction")} />
            </div>
            <ArrowRight className="autograd-stage-arrow" size={19} aria-hidden="true" />
            <div className="autograd-node-column single-column">
              <NodeButton node={leaf("residual")} selected={selectedNode.id === "residual"} active={activeNodeIds.has("residual")} fault={false} onSelect={() => setSelectedNodeId("residual")} />
            </div>
            <ArrowRight className="autograd-stage-arrow" size={19} aria-hidden="true" />
            <div className="autograd-node-column single-column">
              <NodeButton node={leaf("loss")} selected={selectedNode.id === "loss"} active={activeNodeIds.has("loss")} fault={false} onSelect={() => setSelectedNodeId("loss")} />
            </div>
          </div>

          <div className={`autograd-step-explainer status-${currentStep.status}`} aria-live="polite">
            <header><i><PhaseIcon phase={currentStep.phase} /></i><div><span>{phaseLabels[currentStep.phase]}</span><h3>{currentStep.label}</h3></div><code>{visibleStepIndex + 1} / {simulation.steps.length}</code></header>
            <pre><code>{currentStep.call}</code></pre>
            <p>{currentStep.explanation}</p>
            <div className="autograd-step-state">
              <div><span>读取</span>{currentStep.reads.map((item) => <code key={item}>{item}</code>)}</div>
              <ArrowRight size={17} aria-hidden="true" />
              <div><span>写入</span>{currentStep.writes.length ? currentStep.writes.map((item) => <code key={item}>{item}</code>) : <small>没有产生有效状态</small>}</div>
            </div>
            <footer>
              <button type="button" disabled={stepIndex === 0} onClick={() => { setStepIndex(Math.max(0, stepIndex - 1)); setIsPlaying(false); }}><CaretLeft size={15} />上一步</button>
              <strong>{currentStep.status === "fault" ? <><Warning size={15} weight="fill" />错误在这里暴露</> : currentStep.status === "blocked" ? "上游失败，本阶段阻塞" : "本阶段完成"}</strong>
              <button type="button" disabled={stepIndex === simulation.steps.length - 1} onClick={() => { setStepIndex(Math.min(simulation.steps.length - 1, stepIndex + 1)); setIsPlaying(false); }}>下一步<CaretRight size={15} /></button>
            </footer>
          </div>
        </section>

        <aside className="autograd-node-inspector" aria-live="polite">
          <header><div><span>Selected tensor</span><h3>{selectedNode.label}</h3></div><strong>{formatNumber(selectedNode.id === "w" && mode === "in-place" && !mutationReached ? w : selectedNode.value)}</strong></header>
          <pre><code>{selectedNode.expression}</code></pre>
          <dl>
            <div><dt>requires_grad</dt><dd>{String(selectedNode.requiresGrad)}</dd></div>
            <div><dt>is_leaf</dt><dd>{String(selectedNode.isLeaf)}</dd></div>
            <div><dt>grad_fn</dt><dd>{selectedNode.gradFn ?? "None"}</dd></div>
            <div><dt>version</dt><dd>{selectedNode.id === "w" && mode === "in-place" && !mutationReached ? 0 : selectedNode.version ?? "not exposed here"}</dd></div>
            <div><dt>本次上游梯度</dt><dd>{formatNumber(backwardValuesVisible ? selectedNode.backwardGradient : null)}</dd></div>
            <div><dt>Tensor .grad</dt><dd>{formatNumber(storedGradVisible ? selectedNode.storedGrad : null)}</dd></div>
          </dl>
          <section>
            <span>Saved for backward</span>
            {savedValuesVisible && selectedNode.saved.length ? selectedNode.saved.map((item) => <code key={item}>{item}</code>) : <small>{releaseStepIndex !== -1 && visibleStepIndex >= releaseStepIndex ? "Backward 完成后已释放" : "当前阶段没有可读的 saved value"}</small>}
          </section>
          <section className="autograd-contribution-list">
            <span>进入叶子的贡献</span>
            {contributionsVisible && selectedNode.contributions.length ? selectedNode.contributions.map((item) => <div key={item.source}><strong>{item.source}</strong><code>{item.equation} = {formatNumber(item.value)}</code></div>) : <small>反向路径到达叶子后才会出现贡献</small>}
          </section>
          <p>中间 Tensor 默认不会把梯度留在 `.grad`。它们仍会在 backward 中传递上游梯度。</p>
        </aside>
      </div>

      <section className="autograd-branch-merge" aria-labelledby="autograd-merge-title">
        <header><div><span>Gradient accumulation</span><h3 id="autograd-merge-title">同一个参数被多条路径使用，梯度必须在叶子处求和</h3></div><code>AccumulateGrad(w)</code></header>
        <div className="autograd-merge-flow">
          {leaf("w").contributions.length ? leaf("w").contributions.map((item) => (
            <article key={item.source}><span>来自 {item.source}</span><strong>{item.equation}</strong><code>{formatNumber(item.value)}</code></article>
          )) : <article className="is-empty"><span>没有反向贡献</span><strong>{mode === "no-grad" ? "Graph was not recorded" : "Backward aborted"}</strong><code>None</code></article>}
          <ArrowRight size={18} aria-hidden="true" />
          <article className={simulation.finalGradients ? "is-result" : "is-error"}><span>{backwardPasses === 2 ? "Leaf storage after 2 passes" : "Σ contribution"}</span><strong>{simulation.finalGradients ? `${leaf("w").contributions.map((item) => formatNumber(item.value)).join(" + ")}${simulation.gradientMultiplier === 2 ? "，再累加第二轮" : ""}` : "没有有效梯度"}</strong><code>{simulation.finalGradients ? `w.grad = ${formatNumber(simulation.finalGradients.w)}` : "w.grad = None"}</code></article>
        </div>
        {mode === "tracked" && (
          <div className="autograd-iteration-control">
            <div><span>运行多少轮独立 forward + backward</span><button type="button" className={backwardPasses === 1 ? "is-active" : ""} onClick={() => { setBackwardPasses(1); restart(); }}>1 轮</button><button type="button" className={backwardPasses === 2 ? "is-active" : ""} onClick={() => { setBackwardPasses(2); restart(); }}>2 轮</button></div>
            <label><input type="checkbox" checked={zeroBetweenPasses} disabled={backwardPasses === 1} onChange={(event) => { setZeroBetweenPasses(event.target.checked); restart(); }} />第二轮前调用 optimizer.zero_grad()</label>
            <p>{backwardPasses === 1 ? "一张动态图执行一次 backward。" : zeroBetweenPasses ? "图重建两次，第二次开始前清空叶子梯度。" : "图重建两次，但 .grad 没清空，所以继续累加。"}</p>
          </div>
        )}
      </section>

      <section className="autograd-runtime-bridge" aria-labelledby="autograd-runtime-title">
        <header><div><span>Where the work runs</span><h3 id="autograd-runtime-title">Autograd 负责调度依赖，真正的 Tensor 数值计算仍由 CPU 或 GPU Kernel 完成</h3></div><Cpu size={22} weight="duotone" /></header>
        <div className="autograd-runtime-flow">
          <article className={["setup", "forward", "record"].includes(currentStep.phase) ? "is-active" : ""}><Code size={19} weight="duotone" /><span>CPU process</span><strong>Python 调用 Tensor 算子</strong><small>执行实际 control flow</small></article>
          <ArrowRight size={17} aria-hidden="true" />
          <article className={isBackwardPhase ? "is-active" : ""}><Graph size={19} weight="duotone" /><span>Autograd Engine</span><strong>检查依赖并调度 Node</strong><small>Host 侧 C++ engine 与队列</small></article>
          <ArrowRight size={17} aria-hidden="true" />
          <article className={currentStep.phase === "backward" ? "is-active" : ""}><FunctionIcon size={19} weight="duotone" /><span>Backward operator</span><strong>计算局部 VJP</strong><small>例如 mul backward</small></article>
          <ArrowRight size={17} aria-hidden="true" />
          <article className={currentStep.phase === "backward" ? "is-active" : ""}><FlowArrow size={19} weight="duotone" /><span>Dispatcher</span><strong>选择 CPU 或 CUDA 实现</strong><small>由 Tensor device 决定</small></article>
          <ArrowRight size={17} aria-hidden="true" />
          <article className={currentStep.phase === "backward" ? "is-active" : ""}>{device === "cuda" ? <FlowArrow size={19} weight="duotone" /> : <Cpu size={19} weight="duotone" />}<span>{device === "cuda" ? "Compute Stream" : "CPU thread pool"}</span><strong>{device === "cuda" ? "Backward Kernel 排队" : "CPU kernel 被调度"}</strong><small>{device === "cuda" ? "保持 CUDA 任务顺序" : "使用主机执行资源"}</small></article>
          <ArrowRight size={17} aria-hidden="true" />
          <article className={currentStep.phase === "backward" ? "is-active" : ""}>{device === "cuda" ? <Lightning size={19} weight="duotone" /> : <Database size={19} weight="duotone" />}<span>{device === "cuda" ? "GPU SM + HBM" : "CPU cores + DRAM"}</span><strong>Kernel 读写梯度数据</strong><small>{device === "cuda" ? "显存中的真实 bytes" : "主存中的真实 bytes"}</small></article>
        </div>
        <p><strong>当前选择 {device.toUpperCase()}：</strong>计算图的 Node 和依赖由 Autograd Engine 管理；Node 执行时再根据 Tensor device 选择后端。`loss.backward()` 负责发动反向过程，不代表所有梯度数值都在 CPU 上逐元素计算。</p>
      </section>

      <section className="autograd-contracts" aria-labelledby="autograd-contract-title">
        <header><div><span>Engineering contract</span><h3 id="autograd-contract-title">工程里所谓“记录计算图”，记录的是哪些对象</h3></div></header>
        <div>
          <article><Database size={18} weight="duotone" /><span>Tensor storage</span><strong>真正的数值 bytes</strong><p>CPU Tensor 指向主存，CUDA Tensor 指向显存。Tensor 对象本身还持有 shape、dtype、device 等元数据。</p></article>
          <article><BracketsCurly size={18} weight="duotone" /><span>AutogradMeta</span><strong>requires_grad、grad、grad_fn</strong><p>叶子参数把最终结果积累到 `.grad`；非叶子输出通过 `.grad_fn` 指向创建它的 backward Node。</p></article>
          <article><FunctionIcon size={18} weight="duotone" /><span>Backward Node</span><strong>局部 VJP 与 next edges</strong><p>Node 知道怎样把上游梯度变成输入梯度，并通过边把结果送给更早的 Node。</p></article>
          <article><Warning size={18} weight="duotone" /><span>Saved tensor + version</span><strong>只保存求导需要的 forward 值</strong><p>原地写会增加版本号。Backward 读取前检查版本，避免使用已变化的数据算出静默错误的梯度。</p></article>
        </div>
        <p className="autograd-implementation-note"><strong>边界说明：</strong>这张教学图折叠了部分标量运算。真实 `grad_fn` 名称、Node 数量和保存策略属于 PyTorch 实现细节；动态图、链式法则、saved tensor 和版本检查是稳定概念。</p>
      </section>

      <footer className="autograd-reference-footer">
        <span>官方依据</span>
        <a href="https://docs.pytorch.org/docs/stable/notes/autograd.html" target="_blank" rel="noreferrer">PyTorch Autograd mechanics</a>
        <a href="https://docs.pytorch.org/tutorials/beginner/basics/autograd_tutorial.html" target="_blank" rel="noreferrer">Automatic differentiation tutorial</a>
        <a href="https://docs.pytorch.org/docs/stable/generated/torch.no_grad" target="_blank" rel="noreferrer">torch.no_grad reference</a>
      </footer>
    </section>
  );
}
