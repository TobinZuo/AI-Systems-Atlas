import { ArrowCounterClockwise } from "@phosphor-icons/react/ArrowCounterClockwise";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { CheckCircle } from "@phosphor-icons/react/CheckCircle";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Database } from "@phosphor-icons/react/Database";
import { HardDrives } from "@phosphor-icons/react/HardDrives";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { Network } from "@phosphor-icons/react/Network";
import { Stack } from "@phosphor-icons/react/Stack";
import { Terminal } from "@phosphor-icons/react/Terminal";
import { WarningCircle } from "@phosphor-icons/react/WarningCircle";
import { useMemo, useState } from "react";
import {
  collectiveDataPath,
  defaultProcessRankConfig,
  processInitPhases,
  processStatesAtPhase,
  simulateProcessRanks,
  type ClusterLayout,
  type ProcessBackend,
  type ProcessInitFault,
  type ProcessInitPhaseId,
  type ProcessRankConfig,
} from "../playground/processRank";

const faultOptions: Array<{ id: ProcessInitFault; label: string }> = [
  { id: "none", label: "正常初始化" },
  { id: "duplicate-rank", label: "重复 RANK" },
  { id: "world-size-mismatch", label: "WORLD_SIZE 不同" },
  { id: "master-port-mismatch", label: "端口不同" },
  { id: "device-collision", label: "GPU 绑定冲突" },
];

const stateLabels = {
  "not-started": "未启动",
  running: "运行中",
  waiting: "等待成员",
  ready: "通信就绪",
  error: "配置错误",
  blocked: "后续阻塞",
};

export function ProcessRankPlayground() {
  const [config, setConfig] = useState<ProcessRankConfig>(defaultProcessRankConfig);
  const [phaseId, setPhaseId] = useState<ProcessInitPhaseId>("spawn");
  const [selectedRank, setSelectedRank] = useState(0);

  const simulation = useMemo(() => simulateProcessRanks(config), [config]);
  const states = useMemo(() => processStatesAtPhase(simulation, phaseId), [simulation, phaseId]);
  const selectedWorker = simulation.workers[selectedRank];
  const selectedState = states[selectedRank];
  const dataPath = collectiveDataPath(simulation, selectedRank);
  const phaseIndex = processInitPhases.findIndex((phase) => phase.id === phaseId);
  const currentPhase = processInitPhases[phaseIndex];

  const updateConfig = <K extends keyof ProcessRankConfig>(key: K, value: ProcessRankConfig[K]) => {
    setConfig((current) => ({ ...current, [key]: value }));
    setPhaseId("spawn");
  };

  const reset = () => {
    setConfig(defaultProcessRankConfig);
    setPhaseId("spawn");
    setSelectedRank(0);
  };

  const nodes = Array.from(new Set(simulation.workers.map((worker) => worker.hostname)));

  return (
    <section className="rank-playground" id="process-rank-lab" aria-label="Process 与 Rank 交互实验台">
      <header className="rank-playground-header">
        <div><span>Executable distributed runtime</span><h2>四份 `main()`，为什么能加入同一个训练任务</h2><p>逐阶段看每个进程知道什么、在哪里等待，以及真正的数据最终走哪条硬件路径。</p></div>
        <div className="rank-live-facts"><span><small>workers</small><strong>4 OS processes</strong></span><span><small>nodes</small><strong>{nodes.length}</strong></span><span><small>backend</small><strong>{config.backend.toUpperCase()}</strong></span><span><small>selected</small><strong>global rank {selectedRank}</strong></span></div>
        <button type="button" className="rank-reset" onClick={reset}><ArrowCounterClockwise size={15} aria-hidden="true" />重置</button>
      </header>

      <div className="rank-control-deck">
        <fieldset><legend>物理布局</legend><div>{(["single-node", "two-node"] as ClusterLayout[]).map((layout) => <button type="button" className={config.layout === layout ? "is-active" : ""} aria-pressed={config.layout === layout} onClick={() => updateConfig("layout", layout)} key={layout}>{layout === "single-node" ? "1 node × 4 workers" : "2 nodes × 2 workers"}</button>)}</div></fieldset>
        <fieldset><legend>Process Group backend</legend><div>{(["nccl", "gloo"] as ProcessBackend[]).map((backend) => <button type="button" className={config.backend === backend ? "is-active" : ""} aria-pressed={config.backend === backend} onClick={() => updateConfig("backend", backend)} key={backend}>{backend.toUpperCase()}</button>)}</div></fieldset>
        <fieldset className="rank-fault-control"><legend>故障注入</legend><select value={config.fault} onChange={(event) => updateConfig("fault", event.target.value as ProcessInitFault)}>{faultOptions.map((fault) => <option value={fault.id} key={fault.id}>{fault.label}</option>)}</select></fieldset>
      </div>

      <nav className="rank-phase-rail" aria-label="分布式初始化阶段">{processInitPhases.map((phase, index) => <button type="button" className={`${phase.id === phaseId ? "is-active" : ""}${index < phaseIndex ? " is-complete" : ""}`} aria-pressed={phase.id === phaseId} onClick={() => setPhaseId(phase.id)} key={phase.id}><small>0{index + 1}</small><span><strong>{phase.shortLabel}</strong><i>{phase.label}</i></span></button>)}</nav>

      <div className={`rank-phase-summary plane-${currentPhase.plane}`} aria-live="polite"><span>{currentPhase.plane === "launcher" ? <Terminal size={20} weight="duotone" /> : currentPhase.plane === "control" ? <Database size={20} weight="duotone" /> : <Network size={20} weight="duotone" />}</span><div><strong>{currentPhase.label}</strong><small>{currentPhase.description}</small></div><code>{selectedWorker.workerId}: {selectedState.detail}</code></div>

      <section className="rank-cluster-stage" aria-labelledby="rank-cluster-title">
        <header><div><span>Processes, nodes, and devices</span><h3 id="rank-cluster-title">global rank 在整个组里唯一，local rank 只在一台机器内编号</h3></div><code>world_size = {simulation.workers.length}</code></header>
        <div className={`rank-cluster-layout layout-${config.layout}`}>
          {nodes.map((node) => {
            const nodeWorkers = simulation.workers.filter((worker) => worker.hostname === node);
            return <article className="rank-node" key={node}><header><HardDrives size={18} weight="duotone" /><span><strong>{node}</strong><small>{node === "node-0" ? "10.0.0.10 · rendezvous host" : "10.0.0.11"}</small></span><code>node rank {nodeWorkers[0].nodeRank}</code></header><div>{nodeWorkers.map((worker) => {
              const processState = states[worker.globalRank];
              return <button type="button" className={`${selectedRank === worker.globalRank ? "is-selected" : ""} state-${processState.state}`} onClick={() => setSelectedRank(worker.globalRank)} key={worker.workerId}><header><span>PID {worker.pid}</span><strong>Rank {worker.advertisedRank}</strong><i>{stateLabels[processState.state]}</i></header><div className="rank-process-body"><article><Cpu size={16} weight="duotone" /><span>Python worker</span><strong>main()</strong><small>独立虚拟地址空间</small></article><ArrowRight size={14} aria-hidden="true" /><article className="is-device">{config.backend === "nccl" ? <Stack size={16} weight="duotone" /> : <Cpu size={16} weight="duotone" />}<span>{config.backend === "nccl" ? "CUDA device" : "CPU device"}</span><strong>{worker.device}</strong><small>LOCAL_RANK={worker.localRank}</small></article></div><footer><span>global {worker.globalRank}</span><span>local {worker.localRank}</span><span>{worker.hostname}</span></footer></button>;
            })}</div></article>;
          })}
        </div>
        <aside className={`rank-contract-verdict${simulation.fault.failurePhase ? " is-error" : " is-success"}`}>{simulation.fault.failurePhase ? <WarningCircle size={20} weight="fill" /> : <CheckCircle size={20} weight="fill" />}<div><strong>{simulation.fault.title}</strong><span>{simulation.fault.explanation}</span></div><code>{simulation.fault.failurePhase ? `fails at ${simulation.fault.failurePhase}` : "ranks 0,1,2,3 form one world"}</code></aside>
      </section>

      <section className="rank-identity-stage" aria-labelledby="rank-identity-title">
        <header><div><span>Selected process identity</span><h3 id="rank-identity-title">Rank {selectedRank} 的 Python 代码相同，环境变量让它知道自己是谁</h3></div><Terminal size={22} weight="duotone" aria-hidden="true" /></header>
        <div className="rank-identity-layout">
          <div className="rank-env-table"><header><span>{selectedWorker.hostname} / PID {selectedWorker.pid}</span><strong>os.environ</strong></header>{Object.entries(selectedWorker.environment).map(([key, value]) => <div className={key === "RANK" || key === "LOCAL_RANK" ? "is-identity" : key.startsWith("MASTER") ? "is-endpoint" : ""} key={key}><code>{key}</code><strong>{value}</strong><small>{key === "RANK" ? "组内唯一身份" : key === "LOCAL_RANK" ? "本机 device 映射" : key === "WORLD_SIZE" ? "collective 预期参与数" : key === "MASTER_ADDR" || key === "MASTER_PORT" ? "rendezvous 入口" : "本机 worker 数"}</small></div>)}</div>
          <div className="rank-init-code"><span>每个 worker 都执行</span><code><b>torch.cuda.set_device(</b>int(os.environ["LOCAL_RANK"])<b>)</b><br /><br /><b>dist.init_process_group(</b><br />&nbsp;&nbsp;backend="{config.backend}",<br />&nbsp;&nbsp;init_method="env://"<br /><b>)</b><br /><br />rank = dist.get_rank()&nbsp;&nbsp;&nbsp;&nbsp;<i># {selectedWorker.advertisedRank}</i><br />world = dist.get_world_size()&nbsp;<i># {selectedWorker.environment.WORLD_SIZE}</i></code><p><strong>env:// 不是网络协议。</strong>它表示从环境变量读取 Store endpoint、rank 与 world size，再据此初始化默认 Process Group。</p></div>
        </div>
      </section>

      <section className="rank-rendezvous-stage" aria-labelledby="rank-rendezvous-title">
        <header><div><span>Control plane versus data plane</span><h3 id="rank-rendezvous-title">MASTER_PORT 帮大家碰头，不是所有梯度都绕道 rank 0</h3></div><Network size={22} weight="duotone" aria-hidden="true" /></header>
        <div className="rank-plane-grid">
          <article className="rank-control-plane"><header><span>控制面 · initialization</span><strong>少量元数据</strong></header><div className="rank-worker-dots">{simulation.workers.map((worker) => <i className={simulation.fault.affectedWorkerIds.includes(worker.workerId) ? "is-error" : ""} key={worker.workerId}>R{worker.advertisedRank}<small>{worker.environment.MASTER_PORT}</small></i>)}</div><ArrowRight size={18} aria-hidden="true" /><div className="rank-store-node"><Database size={22} weight="duotone" /><span>C10d TCPStore</span><strong>{simulation.rendezvousEndpoint}</strong><small>成员发现、barrier、NCCL unique ID 等初始化信息</small></div><footer>这里识别“谁属于同一任务”</footer></article>
          <article className="rank-data-plane"><header><span>数据面 · collective</span><strong>大量 Tensor bytes</strong></header><div className="rank-data-endpoint"><Database size={19} weight="duotone" /><span>Rank {dataPath.sourceRank} {dataPath.sourceMemory}</span><strong>gradient chunk</strong></div><ArrowRight size={18} aria-hidden="true" /><div className="rank-transport-node"><Network size={22} weight="duotone" /><span>{dataPath.collectiveLayer}</span><strong>{dataPath.transport}</strong><small>{dataPath.crossesNode ? "跨节点经过 NIC 与网络" : "节点内 GPU peer path"}</small></div><ArrowRight size={18} aria-hidden="true" /><div className="rank-data-endpoint"><Database size={19} weight="duotone" /><span>Rank {dataPath.targetRank} {dataPath.targetMemory}</span><strong>receive buffer</strong></div><footer>payload 不使用 MASTER_PORT={selectedWorker.environment.MASTER_PORT}。这里展示常见路径，NCCL 会按实际拓扑与配置选择 transport。</footer></article>
        </div>
      </section>

      <section className="rank-object-stage" aria-labelledby="rank-object-title">
        <header><div><span>Local objects, shared logical membership</span><h3 id="rank-object-title">默认 Process Group 是“每个进程一个对象，同一个逻辑组”</h3></div><Stack size={22} weight="duotone" aria-hidden="true" /></header>
        <div className="rank-object-table"><header><span>OS process</span><span>Python / c10d object</span><span>backend communicator</span><span>逻辑身份</span></header>{simulation.workers.map((worker) => <button type="button" className={worker.globalRank === selectedRank ? "is-active" : ""} onClick={() => setSelectedRank(worker.globalRank)} key={worker.workerId}><span><strong>{worker.workerId}</strong><small>PID {worker.pid}</small></span><code>{worker.defaultGroupObject}</code><code>{worker.communicatorObject}</code><span><strong>{simulation.logicalGroupId}</strong><small>rank {worker.advertisedRank} / {worker.environment.WORLD_SIZE}</small></span></button>)}</div>
        <div className="rank-not-singleton"><WarningCircle size={19} weight="duotone" /><p><strong>不是一个跨进程 singleton。</strong>Python 对象和内存地址无法跨进程直接共享。每个进程独立初始化本地对象，再用相同的 communicator identity、成员数与 rank 映射建立分布式契约。</p></div>
      </section>

      <section className="rank-gpu-stage" aria-labelledby="rank-gpu-title">
        <header><div><span>What actually runs on a GPU</span><h3 id="rank-gpu-title">GPU 上运行的是 Kernel、Block 与 Warp，不是 Python 进程</h3></div><Lightning size={22} weight="duotone" aria-hidden="true" /></header>
        <div className="rank-gpu-flow"><article className="is-process"><Cpu size={20} weight="duotone" /><span>CPU</span><strong>Rank {selectedRank} Python process</strong><small>持有 Tensor metadata、CUDA context、Process Group</small></article><ArrowRight size={18} aria-hidden="true" /><article className="is-stream"><Lightning size={20} weight="duotone" /><span>CUDA Runtime</span><strong>向 Compute / Comm Stream 提交任务</strong><small>CPU enqueue 后，GPU 可异步执行</small></article><ArrowRight size={18} aria-hidden="true" /><article className="is-gpu"><Stack size={20} weight="duotone" /><span>GPU</span><strong>SM 调度 Kernel 的 Thread Block</strong><small>Warp 执行指令，load/store 访问 HBM</small></article><ArrowRight size={18} aria-hidden="true" /><article className="is-memory"><Database size={20} weight="duotone" /><span>{dataPath.sourceMemory}</span><strong>参数、梯度与通信 buffer</strong><small>进程通过 CUDA 地址引用这些 bytes</small></article></div>
        <footer><code>one rank ≈ one CPU process + one CUDA context + one primary GPU</code><p>这是常见的一进程一卡训练拓扑，不是 CUDA 的硬性定义。一个进程可以管理多张 GPU，但 DDP 与 NCCL 通常采用一进程一卡来简化所有权和故障边界。</p></footer>
      </section>

      <section className="rank-sources" aria-labelledby="rank-sources-title"><div><span>Official contracts</span><h3 id="rank-sources-title">页面语义来自 PyTorch 与 NVIDIA 官方文档</h3></div><nav aria-label="Process 与 Rank 官方资料"><a href="https://docs.pytorch.org/docs/stable/distributed.html" target="_blank" rel="noreferrer">torch.distributed 与 init_process_group</a><a href="https://docs.pytorch.org/docs/stable/elastic/run.html" target="_blank" rel="noreferrer">torchrun 环境变量与 rendezvous</a><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html" target="_blank" rel="noreferrer">NCCL communicator 与 unique ID</a><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html" target="_blank" rel="noreferrer">NCCL transport 与 topology</a></nav></section>

      <nav className="rank-bridge-nav" aria-label="Process 与 Rank 相邻专题"><a href="#/gpu/cuda-stream"><Lightning size={17} weight="duotone" /><span><small>下钻</small><strong>Stream 怎样承载 GPU 工作</strong></span></a><a href="#/distributed/ddp"><Network size={17} weight="duotone" /><span><small>下游</small><strong>DDP 怎样消费 Process Group</strong></span></a><a href="#/distributed/zero-1"><Stack size={17} weight="duotone" /><span><small>扩展</small><strong>Rank 怎样成为参数 owner</strong></span></a></nav>
    </section>
  );
}
