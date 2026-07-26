import { ArrowCounterClockwise } from "@phosphor-icons/react/ArrowCounterClockwise";
import { BracketsCurly } from "@phosphor-icons/react/BracketsCurly";
import { CaretLeft } from "@phosphor-icons/react/CaretLeft";
import { CaretRight } from "@phosphor-icons/react/CaretRight";
import { Circuitry } from "@phosphor-icons/react/Circuitry";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Database } from "@phosphor-icons/react/Database";
import { GridFour } from "@phosphor-icons/react/GridFour";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { Memory } from "@phosphor-icons/react/Memory";
import { Pause } from "@phosphor-icons/react/Pause";
import { Play } from "@phosphor-icons/react/Play";
import { SquaresFour } from "@phosphor-icons/react/SquaresFour";
import { Stack } from "@phosphor-icons/react/Stack";
import { useEffect, useMemo, useState, type CSSProperties } from "react";
import {
  CUDA_WARP_SIZE,
  TEACHING_MAX_BLOCKS_PER_SM,
  formatDeviceAddress,
  gpuKernelPhases,
  selectedKernelLane,
  simulateGradientScaleKernel,
  type GPUKernelPhase,
  type KernelLane,
} from "../playground/gpuArchitecture";

const vectorLengths = [48, 70, 96];
const blockSizes = [16, 32, 48, 64];
const smCounts = [2, 4];
const scales = [0.5, 1, 2];

const actorLabels: Record<GPUKernelPhase["actor"], string> = {
  cpu: "CPU 线程",
  runtime: "CUDA Runtime",
  scheduler: "GPU 调度器",
  memory: "Load / Store 单元",
  compute: "CUDA Core",
};

function laneLabel(lane: KernelLane): string {
  if (lane.state === "unused") return "无线程";
  if (lane.state === "guarded") return `i=${lane.globalIndex}`;
  return `g[${lane.globalIndex}]`;
}

function phaseCode(phase: GPUKernelPhase, gridDim: number, blockDim: number, lane: KernelLane): string {
  if (phase.id === "cpu-launch") return "scale_gradient(grad, out, n, scale)";
  if (phase.id === "stream-queue") return `<<<${gridDim}, ${blockDim}, 0, compute_stream>>>`;
  if (phase.id === "grid-expand") return `gridDim.x = ceil(n / ${blockDim}) = ${gridDim}`;
  if (phase.id === "block-schedule") return "Block 整体驻留到一个 SM";
  if (phase.id === "warp-issue") return `Warp = ${CUDA_WARP_SIZE} lanes`;
  if (phase.id === "hbm-read") return lane.inputAddress === null ? "if (i < n) 不成立" : `g = grad[${lane.globalIndex}]`;
  if (phase.id === "register-compute") return lane.inputValue === null ? "该 Lane 被屏蔽" : `y = ${lane.inputValue} × scale`;
  return lane.outputAddress === null ? "没有写回" : `out[${lane.globalIndex}] = ${lane.outputValue}`;
}

function LaneStateLegend() {
  return (
    <div className="gpu-lane-legend" aria-label="Lane 状态图例">
      <span><i className="lane-active" />活跃，执行 load / multiply / store</span>
      <span><i className="lane-guarded" />线程存在，但被 i &lt; n 屏蔽</span>
      <span><i className="lane-unused" />Block 中没有这个线程</span>
    </div>
  );
}

export function GPUArchitecturePlayground() {
  const [vectorLength, setVectorLength] = useState(70);
  const [threadsPerBlock, setThreadsPerBlock] = useState(32);
  const [smCount, setSmCount] = useState(2);
  const [scale, setScale] = useState(0.5);
  const [phaseIndex, setPhaseIndex] = useState(0);
  const [selectedBlockId, setSelectedBlockId] = useState(0);
  const [selectedWarpId, setSelectedWarpId] = useState(0);
  const [selectedLaneId, setSelectedLaneId] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const simulation = useMemo(
    () => simulateGradientScaleKernel({ vectorLength, threadsPerBlock, smCount, scale }),
    [vectorLength, threadsPerBlock, smCount, scale],
  );
  const phase = gpuKernelPhases[phaseIndex];
  const selectedBlock = simulation.blocks[selectedBlockId] ?? simulation.blocks[0];
  const selectedWarp = selectedBlock.warps[selectedWarpId] ?? selectedBlock.warps[0];
  const selectedLane = selectedKernelLane(
    simulation,
    selectedBlock.blockId,
    selectedWarp.warpId,
    selectedLaneId,
  );
  const selectedIndexFormula = selectedLane.threadIdx === null
    ? `Lane ${selectedLane.laneId} 超出 blockDim.x=${simulation.blockDim}`
    : `i = ${selectedBlock.blockId} × ${simulation.blockDim} + ${selectedLane.threadIdx} = ${selectedLane.globalIndex}`;

  useEffect(() => {
    if (selectedBlockId >= simulation.blocks.length) setSelectedBlockId(0);
    if (selectedWarpId >= simulation.warpsPerBlock) setSelectedWarpId(0);
  }, [selectedBlockId, selectedWarpId, simulation.blocks.length, simulation.warpsPerBlock]);

  useEffect(() => {
    if (!isPlaying) return undefined;
    const timer = window.setInterval(() => {
      setPhaseIndex((current) => {
        if (current === gpuKernelPhases.length - 1) {
          setIsPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 1300);
    return () => window.clearInterval(timer);
  }, [isPlaying]);

  const selectPhase = (index: number) => {
    setPhaseIndex(index);
    setIsPlaying(false);
  };

  const togglePlayback = () => {
    if (!isPlaying && phaseIndex === gpuKernelPhases.length - 1) setPhaseIndex(0);
    setIsPlaying((current) => !current);
  };

  const reset = () => {
    setVectorLength(70);
    setThreadsPerBlock(32);
    setSmCount(2);
    setScale(0.5);
    setPhaseIndex(0);
    setSelectedBlockId(0);
    setSelectedWarpId(0);
    setSelectedLaneId(0);
    setIsPlaying(false);
  };

  return (
    <section className="gpu-architecture-lab" id="gpu-architecture-lab" aria-labelledby="gpu-lab-title">
      <header className="gpu-lab-header">
        <div>
          <span>Gradient kernel lab</span>
          <h2 id="gpu-lab-title">一条梯度元素，怎样在 GPU 内部完成计算</h2>
          <p>点击阶段、Block、Warp 和 Lane。所有索引、地址、掩码和结果都来自同一个确定性模拟器。</p>
        </div>
        <pre aria-label="本页模拟的 CUDA kernel"><code>{`__global__ void scale_gradient(float* grad, float* out, int n, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = grad[i] * scale;
}`}</code></pre>
      </header>

      <div className="gpu-lab-controls" aria-label="Kernel launch 配置">
        <label>
          <span>梯度长度 n</span>
          <select value={vectorLength} onChange={(event) => setVectorLength(Number(event.target.value))}>
            {vectorLengths.map((value) => <option value={value} key={value}>{value} elements</option>)}
          </select>
        </label>
        <label>
          <span>blockDim.x</span>
          <select value={threadsPerBlock} onChange={(event) => setThreadsPerBlock(Number(event.target.value))}>
            {blockSizes.map((value) => <option value={value} key={value}>{value} threads</option>)}
          </select>
        </label>
        <label>
          <span>教学 GPU</span>
          <select value={smCount} onChange={(event) => setSmCount(Number(event.target.value))}>
            {smCounts.map((value) => <option value={value} key={value}>{value} SMs</option>)}
          </select>
        </label>
        <label>
          <span>scale</span>
          <select value={scale} onChange={(event) => setScale(Number(event.target.value))}>
            {scales.map((value) => <option value={value} key={value}>× {value}</option>)}
          </select>
        </label>
        <div className="gpu-control-actions">
          <button type="button" className="gpu-play-button" onClick={togglePlayback}>
            {isPlaying ? <Pause size={16} weight="fill" aria-hidden="true" /> : <Play size={16} weight="fill" aria-hidden="true" />}
            {isPlaying ? "暂停" : phaseIndex === gpuKernelPhases.length - 1 ? "重放" : "连续演示"}
          </button>
          <button type="button" className="gpu-reset-button" onClick={reset} aria-label="重置 Kernel 实验">
            <ArrowCounterClockwise size={17} aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="gpu-launch-summary" aria-label="Launch configuration 结果">
        <div><span>Grid</span><strong>{simulation.gridDim} Blocks</strong><code>ceil({vectorLength} / {threadsPerBlock})</code></div>
        <div><span>每个 Block</span><strong>{simulation.warpsPerBlock} Warps</strong><code>ceil({threadsPerBlock} / 32)</code></div>
        <div><span>总 Warp Lane slots</span><strong>{simulation.totalScheduledLanes} Lanes</strong><code>{simulation.totalProgramThreads} CUDA threads</code></div>
        <div><span>有效 Lane</span><strong>{(simulation.usefulLaneRatio * 100).toFixed(1)}%</strong><code>{vectorLength} / {simulation.totalScheduledLanes}</code></div>
      </div>

      <div className="gpu-phase-rail" role="tablist" aria-label="Kernel 执行阶段">
        {gpuKernelPhases.map((item, index) => (
          <button
            type="button"
            role="tab"
            aria-selected={phaseIndex === index}
            className={`${phaseIndex === index ? "is-active" : ""}${index < phaseIndex ? " is-complete" : ""}`}
            onClick={() => selectPhase(index)}
            key={item.id}
          >
            <span>{item.compactLabel}</span>
          </button>
        ))}
      </div>

      <div className={`gpu-operation-band actor-${phase.actor}`}>
        <div className="gpu-operation-icon" aria-hidden="true">
          {phase.actor === "cpu" ? <Cpu size={22} weight="duotone" />
            : phase.actor === "memory" ? <Database size={22} weight="duotone" />
              : phase.actor === "compute" ? <Lightning size={22} weight="duotone" />
                : phase.actor === "scheduler" ? <Circuitry size={22} weight="duotone" />
                  : <BracketsCurly size={22} weight="duotone" />}
        </div>
        <div><span>{actorLabels[phase.actor]}正在工作</span><strong>{phase.label}</strong><p>{phase.explanation}</p></div>
        <code>{phaseCode(phase, simulation.gridDim, simulation.blockDim, selectedLane)}</code>
        <div className="gpu-step-buttons">
          <button type="button" onClick={() => selectPhase(Math.max(0, phaseIndex - 1))} disabled={phaseIndex === 0} aria-label="上一步"><CaretLeft size={17} /></button>
          <span>{phaseIndex + 1} / {gpuKernelPhases.length}</span>
          <button type="button" onClick={() => {
            if (phaseIndex === gpuKernelPhases.length - 1) selectPhase(0);
            else selectPhase(phaseIndex + 1);
          }} aria-label={phaseIndex === gpuKernelPhases.length - 1 ? "重新开始" : "下一步"}><CaretRight size={17} /></button>
        </div>
      </div>

      <div className="gpu-execution-layout">
        <div className="gpu-execution-board">
          <section className={`gpu-hierarchy-row hierarchy-cpu${phaseIndex >= 0 ? " is-reached" : ""}`}>
            <div><Cpu size={18} weight="duotone" /><span>CPU 进程</span></div>
            <code>PyTorch / CUDA Runtime</code>
            <p>准备两个显存指针、n、scale 和 launch configuration，然后把任务提交给 GPU。</p>
          </section>

          <section className={`gpu-hierarchy-row hierarchy-stream${phaseIndex >= 1 ? " is-reached" : ""}`}>
            <div><Stack size={18} weight="duotone" /><span>CUDA Compute Stream</span></div>
            <code>ordered work queue</code>
            <p>同一 Stream 内保持入队顺序。CPU launch 通常不会等待 kernel 完成。</p>
          </section>

          <section className={`gpu-grid-section${phaseIndex >= 2 ? " is-reached" : ""}`}>
            <header><span><GridFour size={18} weight="duotone" />Grid</span><code>{simulation.gridDim} blocks</code></header>
            <div className="gpu-block-queue">
              {simulation.blocks.map((block) => (
                <button
                  type="button"
                  className={`${selectedBlock.blockId === block.blockId ? "is-selected" : ""}${phaseIndex >= 3 ? " is-scheduled" : ""}`}
                  style={{ "--block-sm-color": `var(--gpu-sm-${block.smId % 4})` } as CSSProperties}
                  onClick={() => {
                    setSelectedBlockId(block.blockId);
                    setSelectedWarpId(0);
                  }}
                  key={block.blockId}
                >
                  <strong>B{block.blockId}</strong>
                  <span>{block.activeThreadCount}/{threadsPerBlock} active</span>
                  <small>{phaseIndex >= 3 ? `SM ${block.smId}, wave ${block.wave}` : "等待调度"}</small>
                </button>
              ))}
            </div>
          </section>

          <section className={`gpu-die-section${phaseIndex >= 3 ? " is-reached" : ""}`}>
            <header>
              <span><SquaresFour size={18} weight="duotone" />教学 GPU</span>
              <code>{smCount} SMs, 每个 SM 最多显示 {TEACHING_MAX_BLOCKS_PER_SM} 个 resident blocks</code>
            </header>
            <div className={`gpu-sm-grid sms-${smCount}`}>
              {Array.from({ length: smCount }, (_, smId) => {
                const blocks = simulation.blocks.filter((block) => block.smId === smId);
                return (
                  <article className={`${selectedBlock.smId === smId ? "is-selected" : ""}`} style={{ "--sm-color": `var(--gpu-sm-${smId % 4})` } as CSSProperties} key={smId}>
                    <header><strong>SM {smId}</strong><span>Warp schedulers + cores</span></header>
                    <div className="gpu-resident-blocks">
                      {blocks.map((block) => (
                        <button
                          type="button"
                          className={selectedBlock.blockId === block.blockId ? "is-selected" : ""}
                          onClick={() => {
                            setSelectedBlockId(block.blockId);
                            setSelectedWarpId(0);
                          }}
                          key={block.blockId}
                        >
                          <strong>Block {block.blockId}</strong><span>slot {block.residentSlot}, wave {block.wave}</span>
                        </button>
                      ))}
                    </div>
                  </article>
                );
              })}
            </div>
            <p className="gpu-teaching-assumption"><strong>教学假设：</strong>页面用 round-robin 和每 SM 两个 Block 生成稳定画面。真实 GPU 根据寄存器、共享内存和空闲资源动态调度，不保证 Block 顺序。</p>
          </section>

          <section className={`gpu-warp-section${phaseIndex >= 4 ? " is-reached" : ""}`}>
            <header>
              <div><span><Circuitry size={18} weight="duotone" />Block {selectedBlock.blockId} 内部</span><code>SM {selectedBlock.smId}</code></div>
              <div className="gpu-warp-tabs" role="tablist" aria-label="选择 Warp">
                {selectedBlock.warps.map((warp) => (
                  <button type="button" role="tab" aria-selected={selectedWarp.warpId === warp.warpId} className={selectedWarp.warpId === warp.warpId ? "is-active" : ""} onClick={() => setSelectedWarpId(warp.warpId)} key={warp.warpId}>
                    Warp {warp.warpId} <small>{warp.activeLaneCount}/32 active</small>
                  </button>
                ))}
              </div>
            </header>

            <div className="gpu-lane-grid" aria-label={`Warp ${selectedWarp.warpId} 的 32 个 Lane`}>
              {selectedWarp.lanes.map((lane) => (
                <button
                  type="button"
                  className={`lane-${lane.state}${selectedLane.laneId === lane.laneId ? " is-selected" : ""}`}
                  aria-pressed={selectedLane.laneId === lane.laneId}
                  onClick={() => setSelectedLaneId(lane.laneId)}
                  key={lane.laneId}
                >
                  <strong>{lane.laneId}</strong><span>{laneLabel(lane)}</span>
                </button>
              ))}
            </div>
            <LaneStateLegend />
            <p className="gpu-memory-assumption">32B 事务按 Compute Capability 6.0+ 的简化模型统计。实际流量还取决于地址对齐、缓存命中和具体 GPU 架构。</p>
          </section>
        </div>

        <aside className="gpu-thread-inspector" aria-live="polite">
          <header><span>当前追踪</span><strong>Block {selectedBlock.blockId}, Warp {selectedWarp.warpId}, Lane {selectedLane.laneId}</strong></header>

          <div className={`gpu-thread-verdict state-${selectedLane.state}`}>
            <span>全局索引</span><code>{selectedIndexFormula}</code>
            <strong>
              {selectedLane.state === "active" ? `${selectedLane.globalIndex} < ${vectorLength}，执行 kernel body`
                : selectedLane.state === "guarded" ? `${selectedLane.globalIndex} ≥ ${vectorLength}，被 if guard 屏蔽`
                  : "这个硬件 Lane 没有对应 CUDA thread"}
            </strong>
          </div>

          <div className="gpu-thread-data-path">
            <div className={`${phaseIndex >= 5 && selectedLane.state === "active" ? "is-active" : ""}`}>
              <span><Database size={16} weight="duotone" />Global memory</span>
              <code>{formatDeviceAddress(selectedLane.inputAddress)}</code>
              <strong>{selectedLane.inputValue === null ? "不读取" : `${selectedLane.inputValue}，缓存未命中时访问 HBM`}</strong>
            </div>
            <i aria-hidden="true">↓</i>
            <div className={`${phaseIndex >= 5 && selectedLane.state === "active" ? "is-active" : ""}`}>
              <span><Memory size={16} weight="duotone" />L2 / L1 / LSU</span>
              <code>{selectedLane.inputSegment === null ? "没有事务" : `32B @ ${formatDeviceAddress(selectedLane.inputSegment * 32)}`}</code>
              <strong>{selectedWarp.inputTransactionCount} 个教学事务 / Warp</strong>
            </div>
            <i aria-hidden="true">↓</i>
            <div className={`${phaseIndex >= 6 && selectedLane.state === "active" ? "is-active" : ""}`}>
              <span><Lightning size={16} weight="duotone" />Registers + ALU</span>
              <code>{selectedLane.inputValue === null ? "Lane masked" : `${selectedLane.inputValue} × ${scale}`}</code>
              <strong>{phaseIndex >= 6 && selectedLane.outputValue !== null ? selectedLane.outputValue : "等待执行"}</strong>
            </div>
            <i aria-hidden="true">↓</i>
            <div className={`${phaseIndex >= 7 && selectedLane.state === "active" ? "is-active" : ""}`}>
              <span><Database size={16} weight="duotone" />HBM output</span>
              <code>{formatDeviceAddress(selectedLane.outputAddress)}</code>
              <strong>{phaseIndex >= 7 && selectedLane.outputValue !== null ? selectedLane.outputValue : "尚未写回"}</strong>
            </div>
          </div>

          <dl className="gpu-warp-facts">
            <div><dt>Warp 活跃</dt><dd>{selectedWarp.activeLaneCount} / 32</dd></div>
            <div><dt>Guard 屏蔽</dt><dd>{selectedWarp.guardedLaneCount}</dd></div>
            <div><dt>无对应线程</dt><dd>{selectedWarp.unusedLaneCount}</dd></div>
            <div><dt>共享内存</dt><dd>这个 kernel 未使用</dd></div>
          </dl>
        </aside>
      </div>

      <section className="gpu-concept-contracts" aria-labelledby="gpu-contracts-title">
        <header><h3 id="gpu-contracts-title">哪些是 CUDA 概念，哪些只是这张图的画法</h3></header>
        <div>
          <article><span>CUDA 保证</span><strong>Grid → Block → Thread</strong><p>线程用 blockIdx、blockDim 和 threadIdx 算出全局索引。一个 Block 在一个 SM 上执行。</p></article>
          <article><span>硬件执行</span><strong>Block → Warp → Lane</strong><p>线程按 32 个 Lane 组成 Warp。Warp 是发射指令和合并内存访问的重要观察单位。</p></article>
          <article><span>教学简化</span><strong>Block round-robin → SM</strong><p>真实 Block 调度顺序不可依赖。本页固定分配只是为了让同一配置每次都得到相同画面。</p></article>
        </div>
      </section>

      <section className="gpu-next-connection" aria-labelledby="gpu-next-title">
        <div><span>进入异步执行</span><h3 id="gpu-next-title">这个 kernel 写完梯度，通信才可以安全读取</h3><p>Backward kernel 把 parameter.grad 写入 HBM。下一步观察 DDP hook 怎样把 NCCL collective 提交到 Comm Stream，并用 Event 保证数据就绪。</p></div>
        <a href="#/gpu/cuda-stream">进入 CUDA Stream</a>
      </section>

      <footer className="gpu-reference-footer">
        <strong>官方依据</strong>
        <a href="https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html" target="_blank" rel="noreferrer">CUDA 线程层级与 SIMT kernel</a>
        <a href="https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html" target="_blank" rel="noreferrer">CUDA Stream 与异步执行</a>
        <a href="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory" target="_blank" rel="noreferrer">Global memory coalescing</a>
      </footer>
    </section>
  );
}
