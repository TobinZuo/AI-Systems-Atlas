import { ArrowCounterClockwise } from "@phosphor-icons/react/ArrowCounterClockwise";
import { ArrowDown } from "@phosphor-icons/react/ArrowDown";
import { CaretLeft } from "@phosphor-icons/react/CaretLeft";
import { CaretRight } from "@phosphor-icons/react/CaretRight";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Database } from "@phosphor-icons/react/Database";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { Memory } from "@phosphor-icons/react/Memory";
import { Stack } from "@phosphor-icons/react/Stack";
import { useMemo, useState } from "react";
import {
  MEMORY_DW,
  MEMORY_DY,
  MEMORY_MATRIX_SIZE,
  MEMORY_X,
  effectiveMemoryTileSize,
  gradientEquation,
  gradientTile,
  memoryResidency,
  memorySteps,
  memoryTraffic,
  type MatrixValue,
  type MemoryKernelStrategy,
  type MemoryMatrixId,
} from "../playground/memoryHierarchy";

const memoryLayerFacts = [
  { id: "register", name: "Register file", where: "每个 SM", scope: "每个 thread", owner: "编译器分配", use: "操作数与 accumulator" },
  { id: "shared", name: "Shared Memory", where: "每个 SM", scope: "一个 block", owner: "Kernel 显式寻址", use: "跨 thread 复用 tile" },
  { id: "l1", name: "L1 Cache", where: "每个 SM", scope: "SM 内请求", owner: "硬件自动管理", use: "缓存近期访问" },
  { id: "l2", name: "L2 Cache", where: "整个 GPU", scope: "所有 SM", owner: "硬件自动管理", use: "减少 HBM 访问" },
  { id: "hbm", name: "HBM / Global", where: "GPU Die 之外", scope: "整个 device", owner: "CUDA allocation", use: "保存完整 Tensor" },
];

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

function formatBytes(bytes: number): string {
  return `${bytes} B`;
}

function matrixValue(
  matrix: MemoryMatrixId,
  row: number,
  column: number,
): number {
  if (matrix === "X") return MEMORY_X[row][column];
  if (matrix === "dY") return MEMORY_DY[row][column];
  return MEMORY_DW[row][column];
}

function MatrixGrid({
  matrix,
  selectedRow,
  selectedColumn,
  kValues,
  onSelectOutput,
}: {
  matrix: MemoryMatrixId;
  selectedRow: number;
  selectedColumn: number;
  kValues: number[];
  onSelectOutput: (row: number, column: number) => void;
}) {
  const title = matrix === "X" ? "X" : matrix === "dY" ? "dY" : "dW";
  const role = matrix === "X" ? "activation" : matrix === "dY" ? "upstream gradient" : "weight gradient";
  return (
    <section className={`memory-matrix matrix-${matrix.toLowerCase()}`}>
      <header><strong>{title}</strong><span>{role}</span><code>4 × 4 FP32</code></header>
      <div className="memory-matrix-grid">
        {Array.from({ length: MEMORY_MATRIX_SIZE }, (_, row) =>
          Array.from({ length: MEMORY_MATRIX_SIZE }, (_, column) => {
            const isOperand = matrix === "X"
              ? column === selectedRow
              : matrix === "dY"
                ? column === selectedColumn
                : row === selectedRow && column === selectedColumn;
            const isCurrentK = matrix !== "dW" && kValues.includes(row);
            const className = `${isOperand ? "is-operand" : ""}${isCurrentK ? " is-current-k" : ""}${matrix === "dW" && isOperand ? " is-selected-output" : ""}`;
            const content = <><small>{row},{column}</small><strong>{formatNumber(matrixValue(matrix, row, column))}</strong></>;
            return matrix === "dW" ? (
              <button type="button" className={className} aria-pressed={isOperand} onClick={() => onSelectOutput(row, column)} key={`${row}-${column}`}>{content}</button>
            ) : <span className={className} key={`${row}-${column}`}>{content}</span>;
          }),
        )}
      </div>
    </section>
  );
}

function DataTokens({ values, compact = false }: { values: MatrixValue[]; compact?: boolean }) {
  if (values.length === 0) return <span className="memory-empty-slot">本阶段还没有数据副本</span>;
  return (
    <div className={`memory-data-tokens${compact ? " is-compact" : ""}`}>
      {values.map((value) => (
        <span className={`token-${value.matrix.toLowerCase()}${value.selectedOperand ? " is-selected" : ""}`} key={value.id}>
          <small>{value.id}</small><strong>{formatNumber(value.value)}</strong>
        </span>
      ))}
    </div>
  );
}

export function MemoryHierarchyPlayground() {
  const [strategy, setStrategy] = useState<MemoryKernelStrategy>("tiled");
  const [tileSize, setTileSize] = useState(2);
  const [selectedRow, setSelectedRow] = useState(3);
  const [selectedColumn, setSelectedColumn] = useState(2);
  const [kStart, setKStart] = useState(0);
  const [stepIndex, setStepIndex] = useState(0);

  const effectiveTile = effectiveMemoryTileSize(strategy, tileSize);
  const steps = useMemo(() => memorySteps(strategy), [strategy]);
  const step = steps[stepIndex] ?? steps[0];
  const tile = gradientTile(selectedRow, selectedColumn, kStart, effectiveTile);
  const equation = gradientEquation(selectedRow, selectedColumn);
  const residency = memoryResidency(strategy, step.id);
  const traffic = memoryTraffic(strategy, tileSize);
  const comparisons = [
    memoryTraffic("naive", 1),
    memoryTraffic("tiled", 2),
    memoryTraffic("tiled", 4),
  ];
  const registerValues = [...tile.xValues, ...tile.dyValues].filter((value) => value.selectedOperand);
  const l2Values = residency.l2Tile ? [...tile.xValues, ...tile.dyValues] : [];
  const sharedValues = residency.sharedTile ? [...tile.xValues, ...tile.dyValues] : [];
  const outputValues = residency.hbmOutput ? tile.outputValues : [];
  const maxTrafficBytes = Math.max(...comparisons.map((item) => item.coldPathBytes));

  const selectStrategy = (next: MemoryKernelStrategy) => {
    setStrategy(next);
    setKStart(0);
    setStepIndex(0);
  };

  const selectTileSize = (next: number) => {
    setTileSize(next);
    setKStart(0);
    setStepIndex(0);
  };

  const reset = () => {
    setStrategy("tiled");
    setTileSize(2);
    setSelectedRow(3);
    setSelectedColumn(2);
    setKStart(0);
    setStepIndex(0);
  };

  const selectStep = (index: number) => {
    const nextIndex = Math.max(0, Math.min(steps.length - 1, index));
    if (steps[nextIndex].id === "writeback") {
      setKStart(MEMORY_MATRIX_SIZE - effectiveTile);
    }
    setStepIndex(nextIndex);
  };

  return (
    <section className="memory-hierarchy-lab" id="memory-hierarchy-lab" aria-labelledby="memory-lab-title">
      <header className="memory-lab-header">
        <div>
          <span>Gradient data movement lab</span>
          <h2 id="memory-lab-title">算式没变，性能差别来自同一份数据被搬了几次</h2>
          <p>一个 Linear backward kernel 计算 dW = Xᵀ × dY。点击阶段，观察当前 tile 在每层是否真的存在副本。</p>
        </div>
        <div className="memory-kernel-signature">
          <span>当前 kernel</span><code>dW[m,n] = Σₖ X[k,m] × dY[k,n]</code>
          <small>教学矩阵 4 × 4，FP32，每个标量 4 Bytes</small>
        </div>
        <button type="button" className="memory-reset" onClick={reset} aria-label="重置内存层级实验"><ArrowCounterClockwise size={16} /></button>
      </header>

      <div className="memory-controls">
        <fieldset>
          <legend>Kernel 策略</legend>
          <div>
            <button type="button" className={strategy === "naive" ? "is-active" : ""} aria-pressed={strategy === "naive"} onClick={() => selectStrategy("naive")}><strong>Naive</strong><small>每个输出线程直接 global load</small></button>
            <button type="button" className={strategy === "tiled" ? "is-active" : ""} aria-pressed={strategy === "tiled"} onClick={() => selectStrategy("tiled")}><strong>Tiled</strong><small>Block 先协作装入 Shared</small></button>
          </div>
        </fieldset>
        <fieldset className={strategy === "naive" ? "is-disabled" : ""}>
          <legend>Tile size</legend>
          <div>{[2, 4].map((size) => <button type="button" disabled={strategy === "naive"} className={strategy === "tiled" && tileSize === size ? "is-active" : ""} aria-pressed={strategy === "tiled" && tileSize === size} onClick={() => selectTileSize(size)} key={size}><strong>{size} × {size}</strong><small>一次装入 {2 * size * size} floats</small></button>)}</div>
        </fieldset>
        <fieldset>
          <legend>正在累加的 K slice</legend>
          <div>
            {Array.from({ length: MEMORY_MATRIX_SIZE / effectiveTile }, (_, index) => index * effectiveTile).map((start) => (
              <button type="button" className={kStart === start ? "is-active" : ""} aria-pressed={kStart === start} onClick={() => { setKStart(start); setStepIndex(0); }} key={start}>
                <strong>K {start}..{start + effectiveTile - 1}</strong><small>第 {start / effectiveTile + 1} 个 slice</small>
              </button>
            ))}
          </div>
        </fieldset>
      </div>

      <div className="memory-phase-rail" role="tablist" aria-label="数据移动阶段">
        {steps.map((item, index) => (
          <button type="button" role="tab" aria-selected={index === stepIndex} className={`${index === stepIndex ? "is-active" : ""}${index < stepIndex ? " is-complete" : ""}`} onClick={() => selectStep(index)} key={item.id}>
            <strong>{item.compactLabel}</strong><small>{index + 1}</small>
          </button>
        ))}
      </div>

      <div className={`memory-operation-band layer-${step.activeLayer}`}>
        <div>{step.activeLayer === "hbm" ? <Database size={22} weight="duotone" /> : step.activeLayer === "compute" ? <Lightning size={22} weight="duotone" /> : <Memory size={22} weight="duotone" />}</div>
        <span>当前动作</span><strong>{step.label}</strong><p>{step.explanation}</p>
        <div className="memory-step-actions">
          <button type="button" disabled={stepIndex === 0} onClick={() => selectStep(stepIndex - 1)} aria-label="上一步"><CaretLeft size={16} /></button>
          <code>{stepIndex + 1} / {steps.length}</code>
          <button type="button" disabled={stepIndex === steps.length - 1} onClick={() => selectStep(stepIndex + 1)} aria-label="下一步"><CaretRight size={16} /></button>
        </div>
      </div>

      <div className="memory-main-layout">
        <section className="memory-hardware-stage" aria-labelledby="memory-flow-title">
          <header><div><span>One tile inside one GPU</span><h3 id="memory-flow-title">数据副本现在位于哪一层</h3></div><code>{strategy === "tiled" ? `${effectiveTile} × ${effectiveTile} tile` : "one output thread"}</code></header>

          <div className={`memory-hbm-layer${step.activeLayer === "hbm" ? " is-active" : ""}`}>
            <header><div><Database size={18} weight="duotone" /><strong>HBM</strong><span>global allocations</span></div><code>device scope</code></header>
            <div className="memory-hbm-columns">
              <section><span>输入 Tensor 的当前区域</span><DataTokens values={[...tile.xValues, ...tile.dyValues]} compact /></section>
              <section className={residency.hbmOutput ? "has-output" : ""}><span>dW 输出区域</span><DataTokens values={outputValues} compact /></section>
            </div>
            <p>Global memory 是 CUDA 地址空间。load 先查缓存，只有 miss 才真正走到 HBM。</p>
          </div>

          <ArrowDown className="memory-flow-arrow" size={22} aria-hidden="true" />

          <div className="memory-gpu-chip">
            <header><span><Cpu size={17} weight="duotone" />GPU device</span><code>all SMs share L2</code></header>
            <section className={`memory-l2-layer${residency.l2Tile ? " is-reached" : ""}${step.activeLayer === "l2" ? " is-active" : ""}`}>
              <header><strong>L2 Cache</strong><span>硬件按 cache line 自动填充与淘汰</span><code>GPU scope</code></header>
              <DataTokens values={l2Values} compact />
            </section>

            <ArrowDown className="memory-flow-arrow" size={20} aria-hidden="true" />

            <div className="memory-sm-shell">
              <header><span><Stack size={17} weight="duotone" />SM 0</span><code>one resident thread block</code></header>
              <section className={`memory-shared-layer${residency.sharedTile ? " is-reached" : ""}${step.activeLayer === "shared" ? " is-active" : ""}${strategy === "naive" ? " is-bypassed" : ""}`}>
                <header><strong>Shared Memory</strong><span>{strategy === "tiled" ? "Block 显式管理的 on-chip tile" : "Naive kernel 不分配这块 tile"}</span><code>{strategy === "tiled" ? "block scope" : "bypass"}</code></header>
                {strategy === "tiled" ? <DataTokens values={sharedValues} compact /> : <p className="memory-bypass-note">global load 的结果直接成为线程操作数。L1/L2 仍可能缓存，但 kernel 没有可复用的显式 Shared tile。</p>}
              </section>

              <ArrowDown className="memory-flow-arrow" size={20} aria-hidden="true" />

              <div className="memory-sm-execution">
                <section className={`memory-register-layer${residency.registerOperands ? " is-reached" : ""}${step.activeLayer === "register" ? " is-active" : ""}`}>
                  <header><strong>Thread registers</strong><span>dW[{selectedRow},{selectedColumn}] 的操作数</span></header>
                  <DataTokens values={residency.registerOperands ? registerValues : []} compact />
                </section>
                <section className={`memory-fma-layer${residency.accumulator ? " is-reached" : ""}${step.activeLayer === "compute" ? " is-active" : ""}`}>
                  <header><strong>FMA pipeline</strong><span>register accumulator</span></header>
                  <div><small>进入本 slice 前</small><code>acc = {tile.accumulatorBefore}</code></div>
                  <div><small>当前 slice 贡献</small><code>+ {tile.currentContribution}</code></div>
                  <div><small>完成本 slice 后</small><strong>{residency.accumulator ? `acc = ${tile.accumulatorAfter}` : "等待计算"}</strong></div>
                </section>
              </div>
            </div>
          </div>
        </section>

        <aside className="memory-equation-inspector" aria-live="polite">
          <header><span>Selected output thread</span><strong>dW[{selectedRow},{selectedColumn}] = {equation.result}</strong><p>点击右侧 dW 矩阵中的任意元素，可以换一个 thread 追踪。</p></header>
          <div className="memory-equation-terms">
            {equation.terms.map((term) => {
              const isCurrent = tile.kValues.includes(term.k);
              return <div className={isCurrent ? "is-current" : ""} key={term.k}><small>k = {term.k}</small><code>{term.left} × {term.right}</code><strong>{term.product}</strong></div>;
            })}
          </div>
          <div className="memory-accumulator-equation">
            <span>本次 K slice</span><code>{tile.accumulatorBefore} + ({tile.kValues.map((k) => `${MEMORY_X[k][selectedRow]}×${MEMORY_DY[k][selectedColumn]}`).join(" + ")}) = {tile.accumulatorAfter}</code>
            <small>{tile.accumulatorAfter === equation.result ? "全部 K 已累加，可以写回" : `还需继续处理 K ${kStart + effectiveTile}..${MEMORY_MATRIX_SIZE - 1}`}</small>
          </div>
          <div className="memory-reuse-proof">
            <span>Shared tile 为什么有用</span>
            <strong>{strategy === "tiled" ? `每个 staged scalar 被 ${traffic.reusePerStagedScalar} 个输出线程消费` : "每个输出线程独立发出 global load"}</strong>
            <p>{strategy === "tiled" ? `这批 ${tile.xValues.length + tile.dyValues.length} 个输入支持 ${tile.outputValues.length} 个 dW 输出的本轮乘加。` : "硬件 cache 可能碰巧命中，但 kernel 没有声明跨线程复用结构。"}</p>
          </div>
        </aside>
      </div>

      <section className="memory-matrix-workbench" aria-labelledby="memory-matrix-title">
        <header><div><span>Exact tensors in HBM</span><h3 id="memory-matrix-title">选一个 dW 元素，反向定位它读取的 X 列与 dY 列</h3></div><code>dW = Xᵀ × dY</code></header>
        <div className="memory-matrix-equation">
          <MatrixGrid matrix="X" selectedRow={selectedRow} selectedColumn={selectedColumn} kValues={tile.kValues} onSelectOutput={() => {}} />
          <span>ᵀ</span><i>×</i>
          <MatrixGrid matrix="dY" selectedRow={selectedRow} selectedColumn={selectedColumn} kValues={tile.kValues} onSelectOutput={() => {}} />
          <i>=</i>
          <MatrixGrid matrix="dW" selectedRow={selectedRow} selectedColumn={selectedColumn} kValues={tile.kValues} onSelectOutput={(row, column) => { setSelectedRow(row); setSelectedColumn(column); setStepIndex(0); }} />
        </div>
        <footer><span className="legend-x">X operand</span><span className="legend-dy">dY operand</span><span className="legend-k">current K slice</span><span className="legend-output">selected dW</span></footer>
      </section>

      <section className="memory-traffic-stage" aria-labelledby="memory-traffic-title">
        <header><div><span>Data movement ledger</span><h3 id="memory-traffic-title">三种 kernel 做 64 次 FMA，但发出的 global loads 不同</h3><p>下面使用冷缓存教学假设：每个 global load 都需要从 HBM 取数。真实 HBM 流量还会受 L1/L2 命中、cache line、写策略与具体架构影响。</p></div><code>4 × 4 × 4 = 64 FMA</code></header>
        <div className="memory-traffic-cards">
          {comparisons.map((item) => {
            const active = item.strategy === strategy && (strategy === "naive" || item.tileSize === tileSize);
            return (
              <button type="button" className={active ? "is-active" : ""} aria-pressed={active} onClick={() => {
                if (item.strategy === "naive") selectStrategy("naive");
                else {
                  setStrategy("tiled");
                  selectTileSize(item.tileSize);
                }
              }} key={`${item.strategy}-${item.tileSize}`}>
                <header><strong>{item.strategy === "naive" ? "Naive" : `Tile ${item.tileSize} × ${item.tileSize}`}</strong><span>{item.reusePerStagedScalar}× explicit reuse</span></header>
                <div><span>Global reads</span><strong>{item.globalReadScalars} floats</strong></div>
                <div><span>Output writes</span><strong>{item.globalWriteScalars} floats</strong></div>
                <div><span>Cold path traffic</span><strong>{formatBytes(item.coldPathBytes)}</strong></div>
                <div><span>Arithmetic intensity</span><strong>{item.arithmeticIntensity} FLOP/B</strong></div>
                <i><span style={{ width: `${item.coldPathBytes / maxTrafficBytes * 100}%` }} /></i>
              </button>
            );
          })}
        </div>
        <div className="memory-traffic-formula">
          <div><span>Naive global reads</span><code>2 × M × N × K = 128 floats</code></div>
          <div><span>Tiled global reads</span><code>2 × M × N × K / tile = {traffic.globalReadScalars} floats</code></div>
          <div><span>没变的计算</span><code>2 × M × N × K = {traffic.flops} FLOPs</code></div>
        </div>
      </section>

      <section className="memory-layer-truth" aria-labelledby="memory-truth-title">
        <header><div><span>Programming model vs physical storage</span><h3 id="memory-truth-title">SRAM 不是一个单独的 CUDA API 层</h3><p>Cache、Shared Memory 和 register file 都位于片上，但可见范围与控制方式不同。Global memory 则是地址空间语义，不等于每次访问都直达 HBM。</p></div></header>
        <div>
          {memoryLayerFacts.map((layer) => <article className={`layer-${layer.id}`} key={layer.id}><header><strong>{layer.name}</strong><span>{layer.where}</span></header><dl><div><dt>谁能看见</dt><dd>{layer.scope}</dd></div><div><dt>谁来管理</dt><dd>{layer.owner}</dd></div><div><dt>本例用途</dt><dd>{layer.use}</dd></div></dl></article>)}
        </div>
        <aside><Memory size={20} weight="duotone" /><p><strong>L1 与 Shared 的关系依架构而异：</strong>现代 NVIDIA GPU 的 SM unified data cache 提供 L1 与 Shared 的物理资源，两者配额可配置。页面画成两个逻辑区域，是为了区分“硬件自动缓存”和“kernel 显式 scratchpad”。</p></aside>
      </section>

      <section className="memory-sources" aria-labelledby="memory-sources-title">
        <header><span>Primary references</span><h3 id="memory-sources-title">页面语义依据</h3></header>
        <div>
          <a href="https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html" target="_blank" rel="noreferrer"><strong>CUDA Programming Guide</strong><span>SM、register、Shared、L1 与 L2 的作用域和物理关系</span></a>
          <a href="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab" target="_blank" rel="noreferrer"><strong>CUDA Best Practices Guide</strong><span>用 Shared Memory 复用矩阵 tile，减少冗余 global-memory transfer</span></a>
        </div>
      </section>
    </section>
  );
}
