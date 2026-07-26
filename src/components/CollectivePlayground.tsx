import { ArrowCounterClockwise } from "@phosphor-icons/react/ArrowCounterClockwise";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { CheckCircle } from "@phosphor-icons/react/CheckCircle";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Database } from "@phosphor-icons/react/Database";
import { Function as FunctionIcon } from "@phosphor-icons/react/Function";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { Network } from "@phosphor-icons/react/Network";
import { Stack } from "@phosphor-icons/react/Stack";
import { WarningCircle } from "@phosphor-icons/react/WarningCircle";
import { useMemo, useState } from "react";
import {
  collectiveCallPacket,
  collectiveOperationDefinitions,
  defaultCollectiveConfig,
  rankViewAtPhase,
  simulateCollective,
  type CollectiveBackend,
  type CollectiveConfig,
  type CollectiveFault,
  type CollectiveOperation,
  type CollectivePhase,
  type CollectiveReduction,
  type CollectiveSlot,
} from "../playground/collectives";

const phaseDefinitions: Array<{ id: CollectivePhase; label: string; detail: string }> = [
  { id: "inputs", label: "Inputs ready", detail: "每个 rank 的本地 Tensor 已在内存" },
  { id: "contract", label: "Calls match", detail: "Backend 对齐 operation、count、dtype 与 root" },
  { id: "transfer", label: "Transfer", detail: "部分 bytes 已经写入目标槽位" },
  { id: "complete", label: "Outputs ready", detail: "Collective 契约承诺的结果全部可见" },
];

const faultOptions: Array<{ id: CollectiveFault; label: string }> = [
  { id: "none", label: "正常契约" },
  { id: "missing-rank", label: "Rank 3 缺席" },
  { id: "operation-mismatch", label: "Operation 不同" },
  { id: "count-mismatch", label: "Count 不同" },
  { id: "root-mismatch", label: "Root 不同" },
];

function formatValues(values: number[] | null): string {
  return values ? `[${values.join(", ")}]` : "空";
}

function slotStateLabel(slot: CollectiveSlot): string {
  if (slot.state === "pending") return "尚未写入";
  if (slot.state === "partial") return "部分结果";
  if (slot.state === "not-applicable") return "本 rank 无输出";
  if (slot.state === "coordination") return "同步完成";
  if (slot.state === "resident") return "本地已有";
  return "接收完成";
}

export function CollectivePlayground() {
  const [config, setConfig] = useState<CollectiveConfig>(defaultCollectiveConfig);
  const [phase, setPhase] = useState<CollectivePhase>("transfer");
  const [selectedRank, setSelectedRank] = useState(0);

  const simulation = useMemo(() => simulateCollective(config), [config]);
  const views = useMemo(() => simulation.ranks.map((rank) => rankViewAtPhase(simulation, rank.rank, phase)), [simulation, phase]);
  const selectedView = views[selectedRank];
  const packet = useMemo(() => collectiveCallPacket(simulation, selectedRank), [simulation, selectedRank]);
  const phaseIndex = phaseDefinitions.findIndex((item) => item.id === phase);
  const selectedRoutes = simulation.routes.filter((route) => route.from === selectedRank || route.to === selectedRank);

  const updateConfig = <K extends keyof CollectiveConfig>(key: K, value: CollectiveConfig[K]) => {
    setConfig((current) => ({ ...current, [key]: value }));
  };

  const selectOperation = (operation: CollectiveOperation) => {
    const definition = collectiveOperationDefinitions.find((item) => item.id === operation)!;
    setConfig((current) => ({ ...current, operation, fault: current.fault === "root-mismatch" && !definition.rootRequired ? "none" : current.fault }));
    setPhase("transfer");
  };

  const reset = () => {
    setConfig(defaultCollectiveConfig);
    setPhase("transfer");
    setSelectedRank(0);
  };

  return (
    <section className="collective-playground" id="collective-lab" aria-label="Collective 交互实验台">
      <header className="collective-playground-header"><div><span>Executable collective semantics</span><h2>同一组 Rank，九种完全不同的数据契约</h2><p>所有数值都由当前 operation、root 和 reduce op 实时计算。</p></div><div className="collective-live-facts"><span><small>operation</small><strong>{simulation.definition.label}</strong></span><span><small>family</small><strong>{simulation.definition.family}</strong></span><span><small>selected</small><strong>rank {selectedRank}</strong></span><span><small>backend</small><strong>{config.backend.toUpperCase()}</strong></span></div><button type="button" className="collective-reset" onClick={reset}><ArrowCounterClockwise size={15} aria-hidden="true" />重置</button></header>

      <div className="collective-operation-deck" role="group" aria-label="选择 Collective operation">{collectiveOperationDefinitions.map((definition) => <button type="button" className={`${config.operation === definition.id ? "is-active" : ""} family-${definition.family}`} aria-pressed={config.operation === definition.id} onClick={() => selectOperation(definition.id)} key={definition.id}><strong>{definition.label}</strong><small>{definition.outputDescription}</small></button>)}</div>

      <div className="collective-control-deck">
        <fieldset><legend>Backend</legend><div>{(["nccl", "gloo"] as CollectiveBackend[]).map((backend) => <button type="button" className={config.backend === backend ? "is-active" : ""} aria-pressed={config.backend === backend} onClick={() => updateConfig("backend", backend)} key={backend}>{backend.toUpperCase()}</button>)}</div></fieldset>
        <fieldset disabled={!simulation.definition.rootRequired}><legend>Root rank {simulation.definition.rootRequired ? "" : "· not used"}</legend><div>{[0, 1, 2, 3].map((rank) => <button type="button" className={config.rootRank === rank ? "is-active" : ""} aria-pressed={config.rootRank === rank} onClick={() => updateConfig("rootRank", rank)} key={rank}>R{rank}</button>)}</div></fieldset>
        <fieldset disabled={!simulation.definition.reductionRequired}><legend>Reduce op {simulation.definition.reductionRequired ? "" : "· not used"}</legend><div>{(["sum", "avg", "max"] as CollectiveReduction[]).map((reduction) => <button type="button" className={config.reduction === reduction ? "is-active" : ""} aria-pressed={config.reduction === reduction} onClick={() => updateConfig("reduction", reduction)} key={reduction}>{reduction.toUpperCase()}</button>)}</div></fieldset>
        <fieldset className="collective-fault-control"><legend>Contract fault</legend><select value={config.fault} onChange={(event) => updateConfig("fault", event.target.value as CollectiveFault)}>{faultOptions.map((fault) => <option value={fault.id} disabled={fault.id === "root-mismatch" && !simulation.definition.rootRequired} key={fault.id}>{fault.label}</option>)}</select></fieldset>
      </div>

      <nav className="collective-phase-rail" aria-label="Collective 生命周期">{phaseDefinitions.map((item, index) => <button type="button" className={`${phase === item.id ? "is-active" : ""}${index < phaseIndex ? " is-complete" : ""}`} aria-pressed={phase === item.id} onClick={() => setPhase(item.id)} key={item.id}><small>0{index + 1}</small><span><strong>{item.label}</strong><i>{item.detail}</i></span></button>)}</nav>

      <div className="collective-contract-band"><Network size={21} weight="duotone" aria-hidden="true" /><div><strong>{simulation.definition.inputDescription}</strong><span>{simulation.definition.outputDescription}</span></div><code>{simulation.definition.commonUse}</code></div>

      <section className="collective-rank-stage" aria-labelledby="collective-rank-title"><header><div><span>Input buffers to output buffers</span><h3 id="collective-rank-title">空槽不是丢数据，而是接收 buffer 的这个位置还没被写入</h3></div><code>{phaseDefinitions[phaseIndex].label}</code></header><div className="collective-rank-grid">{views.map((view) => {
        const affected = simulation.fault.affectedRanks.includes(view.rank);
        return <button type="button" className={`${selectedRank === view.rank ? "is-selected" : ""}${affected ? " is-affected" : ""}${!view.participates ? " is-missing" : ""}`} onClick={() => setSelectedRank(view.rank)} key={view.rank}><header><span>Rank {view.rank}</span><strong>{view.participates ? view.callName : "没有调用"}</strong><small>count={view.count}</small></header><div className="collective-buffer-column"><span>INPUT / SEND BUFFER</span><div>{view.inputSlots.map((slot) => <article className={`state-${slot.state}`} key={`${slot.slot}-${slot.label}`}><small>{slot.label}</small><strong>{formatValues(slot.values)}</strong><i>{slotStateLabel(slot)}</i></article>)}</div></div><ArrowRight size={16} aria-hidden="true" /><div className="collective-buffer-column is-output"><span>OUTPUT / RECEIVE BUFFER</span><div>{view.outputSlots.map((slot) => <article className={`state-${slot.state}`} key={`${slot.slot}-${slot.label}`}><small>{slot.label}</small><strong>{formatValues(slot.values)}</strong><i>{slotStateLabel(slot)}</i></article>)}</div></div><footer>{selectedRank === view.rank ? "正在追踪此 rank" : `点击追踪 Rank ${view.rank}`}</footer></button>;
      })}</div><div className="collective-empty-explainer"><Database size={19} weight="duotone" /><p><strong>{phase === "transfer" ? "为什么现在还有空槽？" : phase === "complete" ? "最终状态是否应该有空槽？" : "输出 buffer 什么时候出现值？"}</strong>{phase === "transfer" ? `当前只演示逻辑构建中的局部进度。以 Rank ${selectedRank} 为例，本地已有的 slot 可以先显示，远端 slot 在收到对应字节前保持空。` : phase === "complete" ? (simulation.fault.failurePhase ? "调用契约不一致，Collective 无法完成，因此输出仍然 pending。" : "Collective 完成后，语义要求有结果的 slot 都已写入；not applicable 表示该操作本来就不向这个 rank 返回结果。") : "Inputs ready 和 Calls match 阶段还没有把远端 Tensor bytes 写入 receive buffer。"}</p></div></section>

      <section className="collective-route-stage" aria-labelledby="collective-route-title"><header><div><span>Logical dependencies, not transport schedule</span><h3 id="collective-route-title">Rank {selectedRank} 需要和谁交换哪些逻辑数据</h3></div><Stack size={22} weight="duotone" aria-hidden="true" /></header><div className="collective-route-layout"><article className="collective-selected-rank"><span>Selected participant</span><strong>Rank {selectedRank}</strong><code>{selectedView.callName}</code><small>{selectedRoutes.filter((route) => route.from === selectedRank).length} outgoing · {selectedRoutes.filter((route) => route.to === selectedRank).length} incoming dependencies</small></article><div className="collective-route-list">{selectedRoutes.slice(0, 12).map((route, index) => <span className={`action-${route.action}`} key={`${route.from}-${route.to}-${route.slot}-${index}`}><b>R{route.from}</b><ArrowRight size={13} aria-hidden="true" /><b>R{route.to}</b><code>{route.label}</code><small>{route.action}</small></span>)}</div></div><footer><WarningCircle size={18} weight="duotone" /><p><strong>这不是 Ring 或 Tree 的逐轮时间表。</strong>它只画 Collective 语义要求的数据依赖。NCCL 会根据消息大小、拓扑与配置选择算法，实际传输可以分块、流水化，并与这些逻辑边不同。</p></footer></section>

      <section className="collective-packet-stage" aria-labelledby="collective-packet-title"><header><div><span>Framework metadata to backend call</span><h3 id="collective-packet-title">通信库不会猜 Tensor 格式，框架把地址和解释方式一起传下去</h3></div><FunctionIcon size={22} weight="duotone" aria-hidden="true" /></header><div className="collective-layer-flow"><article className="is-python"><Cpu size={20} weight="duotone" /><span>Rank {selectedRank} Python</span><strong>{packet.pythonApi}</strong><small>Tensor 对象知道 shape、dtype、device 与 storage</small></article><ArrowRight size={18} aria-hidden="true" /><article className="is-process-group"><Network size={20} weight="duotone" /><span>Process Group</span><strong>选择 {config.backend.toUpperCase()} backend</strong><small>检查 group、root、reduce op 与异步语义</small></article><ArrowRight size={18} aria-hidden="true" /><article className="is-backend"><Lightning size={20} weight="duotone" /><span>Backend API</span><strong>{packet.backendApi}</strong><small>{packet.stream}</small></article><ArrowRight size={18} aria-hidden="true" /><article className="is-buffer"><Database size={20} weight="duotone" /><span>{config.backend === "nccl" ? "GPU HBM" : "CPU DRAM"}</span><strong>直接读写 buffer bytes</strong><small>结果写进 receive pointer 指向的地址</small></article></div><div className="collective-call-packet"><header><span>Rank {selectedRank} call packet</span><strong>{packet.backendApi}</strong></header><dl><div><dt>sendbuff</dt><dd>{packet.sendPointer}</dd></div><div><dt>recvbuff</dt><dd>{packet.receivePointer}</dd></div><div><dt>count</dt><dd>{packet.count} elements</dd></div><div><dt>datatype</dt><dd>{packet.dtype}</dd></div><div><dt>reduce op</dt><dd>{packet.reduceOp}</dd></div><div><dt>root</dt><dd>{packet.root}</dd></div><div><dt>communicator</dt><dd>{packet.communicator}</dd></div><div><dt>execution</dt><dd>{packet.stream}</dd></div></dl><footer><code>{packet.inPlace ? "send pointer == receive pointer · in-place" : "send pointer != receive pointer · out-of-place"}</code><p>传给 Backend 的不是“只有地址”。地址只说明 bytes 在哪里；count、datatype、operation、communicator 和 stream 才说明怎样解释、和谁通信以及写到哪里。</p></footer></div></section>

      <section className="collective-fault-stage" aria-labelledby="collective-fault-title"><header><div><span>Every rank must enter the same contract</span><h3 id="collective-fault-title">Collective 是多人会合点，少一个或顺序不同都会破坏整组进度</h3></div>{simulation.fault.failurePhase ? <WarningCircle size={22} weight="fill" /> : <CheckCircle size={22} weight="fill" />}</header><div className="collective-call-grid">{simulation.ranks.map((rank) => <article className={`${simulation.fault.affectedRanks.includes(rank.rank) ? "is-error" : ""}${!rank.participates ? " is-missing" : ""}`} key={rank.rank}><span>Rank {rank.rank}</span><strong>{rank.participates ? rank.callName : "NO CALL"}</strong><code>count={rank.count} · {config.backend}</code><small>{simulation.definition.rootRequired ? `root=${rank.rank === 3 && config.fault === "root-mismatch" ? (config.rootRank + 1) % 4 : config.rootRank}` : "no root"}</small></article>)}</div><aside className={simulation.fault.failurePhase ? "is-error" : "is-success"}>{simulation.fault.failurePhase ? <WarningCircle size={19} weight="fill" /> : <CheckCircle size={19} weight="fill" />}<div><strong>{simulation.fault.title}</strong><span>{simulation.fault.explanation}</span></div><code>{simulation.fault.failurePhase ? "outputs cannot commit" : "contract accepted"}</code></aside></section>

      <section className="collective-semantics-stage" aria-labelledby="collective-semantics-title"><header><div><span>Contract versus algorithm versus transport</span><h3 id="collective-semantics-title">AllReduce 是结果语义，Ring 只是实现它的一种调度</h3></div><Network size={22} weight="duotone" aria-hidden="true" /></header><div className="collective-three-layers"><article><span>01 · API contract</span><strong>{simulation.definition.label}</strong><p>{simulation.definition.outputDescription}</p><code>由应用选择，结果必须遵守</code></article><ArrowRight size={18} aria-hidden="true" /><article><span>02 · Communication algorithm</span><strong>{config.operation === "all-reduce" ? "Ring / Tree / NVLS ..." : "Backend-selected schedule"}</strong><p>{config.operation === "all-reduce" ? "常见 Ring 可分解为 ReduceScatter + AllGather。" : "分块、路由和轮次不由 API 名称唯一决定。"}</p><code>由 Backend 按拓扑与消息选择</code></article><ArrowRight size={18} aria-hidden="true" /><article><span>03 · Physical transport</span><strong>{config.backend === "nccl" ? "NVLink / PCIe / RDMA / Socket" : "CPU memory + network"}</strong><p>搬运地址指向的真实 Tensor bytes，并在需要时执行归约。</p><code>由硬件拓扑和系统配置约束</code></article></div><div className="collective-allreduce-identity"><span>重要恒等式</span><strong>AllReduce</strong><i>=</i><strong>ReduceScatter</strong><i>+</i><strong>AllGather</strong><small>这是结果与常见分解的等价关系，不表示所有实现都必须使用同一 Ring 时间表。</small></div></section>

      <section className="collective-atlas-stage" aria-labelledby="collective-atlas-title"><header><div><span>Nine operations at one glance</span><h3 id="collective-atlas-title">先看“谁有输入、谁有输出”，再记 API 名称</h3></div><Stack size={22} weight="duotone" aria-hidden="true" /></header><div className="collective-atlas-table"><header><span>Operation</span><span>输入拓扑</span><span>输出拓扑</span><span>典型用途</span></header>{collectiveOperationDefinitions.map((definition) => <button type="button" className={definition.id === config.operation ? "is-active" : ""} onClick={() => selectOperation(definition.id)} key={definition.id}><strong>{definition.label}</strong><span>{definition.inputDescription}</span><span>{definition.outputDescription}</span><span>{definition.commonUse}</span></button>)}</div></section>

      <section className="collective-sources" aria-labelledby="collective-sources-title"><div><span>Official contracts</span><h3 id="collective-sources-title">操作语义和调用约束来自 PyTorch 与 NVIDIA 官方文档</h3></div><nav aria-label="Collective 官方资料"><a href="https://docs.pytorch.org/docs/stable/distributed.html" target="_blank" rel="noreferrer">PyTorch distributed collectives</a><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html" target="_blank" rel="noreferrer">NCCL Collective Operations</a><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html" target="_blank" rel="noreferrer">NCCL algorithms and transports</a><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html" target="_blank" rel="noreferrer">Communicator rank contract</a></nav></section>

      <nav className="collective-bridge-nav" aria-label="Collective 相邻专题"><a href="#/distributed/process-rank"><Cpu size={17} weight="duotone" /><span><small>上游</small><strong>Rank 与 Process Group 怎样建立</strong></span></a><a href="#/distributed/ddp"><Network size={17} weight="duotone" /><span><small>下游</small><strong>DDP 怎样用 AllReduce</strong></span></a><a href="#/distributed/fsdp"><Stack size={17} weight="duotone" /><span><small>扩展</small><strong>FSDP 怎样组合 Gather 与 Scatter</strong></span></a></nav>
    </section>
  );
}
