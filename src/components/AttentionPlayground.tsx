import { ArrowCounterClockwise } from "@phosphor-icons/react/ArrowCounterClockwise";
import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { CaretLeft } from "@phosphor-icons/react/CaretLeft";
import { CaretRight } from "@phosphor-icons/react/CaretRight";
import { Eye } from "@phosphor-icons/react/Eye";
import { Function as FunctionIcon } from "@phosphor-icons/react/Function";
import { GridFour } from "@phosphor-icons/react/GridFour";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { Stack } from "@phosphor-icons/react/Stack";
import { useMemo, useState } from "react";
import {
  ATTENTION_TOKENS,
  ATTENTION_WK,
  ATTENTION_WQ,
  ATTENTION_WV,
  attentionCost,
  attentionHeadLayout,
  attentionMatrixForPhase,
  attentionPhases,
  createAttentionSimulation,
  selectedAttentionDetail,
  type AttentionMaskMode,
  type AttentionScaleMode,
} from "../playground/attention";

function formatNumber(value: number, digits = 2): string {
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

function formatVector(values: number[]): string {
  return `[${values.map((value) => formatNumber(value)).join(", ")}]`;
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatBytes(bytes: number): string {
  const units = ["B", "KiB", "MiB", "GiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

function formatCount(value: number): string {
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(value);
}

function matrixText(value: number | null, phaseIndex: number): string {
  if (phaseIndex < 2) return "等待";
  if (value === null) return "−∞";
  if (phaseIndex >= 5) return formatPercent(value);
  return formatNumber(value);
}

function projectionLabel(matrix: number[][]): string {
  return matrix.map((row) => `[${row.join(" ")}]`).join(" ");
}

export function AttentionPlayground() {
  const [maskMode, setMaskMode] = useState<AttentionMaskMode>("causal");
  const [scaleMode, setScaleMode] = useState<AttentionScaleMode>("scaled");
  const [phaseIndex, setPhaseIndex] = useState(2);
  const [selectedQuery, setSelectedQuery] = useState(2);
  const [selectedKey, setSelectedKey] = useState(0);
  const [headCount, setHeadCount] = useState(2);

  const simulation = useMemo(
    () => createAttentionSimulation(maskMode, scaleMode),
    [maskMode, scaleMode],
  );
  const phase = attentionPhases[phaseIndex];
  const matrix = attentionMatrixForPhase(simulation, phase.id);
  const detail = selectedAttentionDetail(simulation, selectedQuery, selectedKey);
  const headLayout = attentionHeadLayout(8, headCount, ATTENTION_TOKENS.length);
  const costExamples = [128, 1024, 8192].map((length) => attentionCost(length, 32, 128, 2));
  const maxScoreBytes = Math.max(...costExamples.map((cost) => cost.scoreBytes));

  const reset = () => {
    setMaskMode("causal");
    setScaleMode("scaled");
    setPhaseIndex(2);
    setSelectedQuery(2);
    setSelectedKey(0);
    setHeadCount(2);
  };

  return (
    <section className="attention-lab" id="attention-lab" aria-labelledby="attention-lab-title">
      <header className="attention-lab-header">
        <div>
          <span>Scaled dot-product attention lab</span>
          <h2 id="attention-lab-title">固定一个 Query，看它怎样从四个 Value 取回信息</h2>
          <p>每个分数、Mask、概率和输出向量都来自同一份 4-token 数值模型。页面不自动播放，点击阶段逐步观察。</p>
        </div>
        <div className="attention-formula-card">
          <span>核心公式</span><code>Attention(Q,K,V) = softmax(QKᵀ / √dₖ + mask)V</code>
          <small>当前 head: T=4, dₖ={simulation.headDim}, dᵥ={simulation.valueDim}</small>
        </div>
        <button type="button" className="attention-reset" onClick={reset} aria-label="重置 Attention 实验"><ArrowCounterClockwise size={16} /></button>
      </header>

      <div className="attention-controls">
        <fieldset>
          <legend>可见范围</legend>
          <div>
            <button type="button" className={maskMode === "causal" ? "is-active" : ""} aria-pressed={maskMode === "causal"} onClick={() => { setMaskMode("causal"); setPhaseIndex(4); }}><strong>Causal</strong><small>只能看自己和过去</small></button>
            <button type="button" className={maskMode === "bidirectional" ? "is-active" : ""} aria-pressed={maskMode === "bidirectional"} onClick={() => { setMaskMode("bidirectional"); setPhaseIndex(4); }}><strong>Bidirectional</strong><small>可以看全部 token</small></button>
          </div>
        </fieldset>
        <fieldset>
          <legend>点积缩放</legend>
          <div>
            <button type="button" className={scaleMode === "scaled" ? "is-active" : ""} aria-pressed={scaleMode === "scaled"} onClick={() => { setScaleMode("scaled"); setPhaseIndex(3); }}><strong>÷ √dₖ</strong><small>标准 scaled attention</small></button>
            <button type="button" className={scaleMode === "unscaled" ? "is-active" : ""} aria-pressed={scaleMode === "unscaled"} onClick={() => { setScaleMode("unscaled"); setPhaseIndex(3); }}><strong>不缩放</strong><small>比较 Softmax 集中程度</small></button>
          </div>
        </fieldset>
        <fieldset>
          <legend>追踪哪个 Query</legend>
          <div>{simulation.tokens.map((token, index) => <button type="button" className={selectedQuery === index ? "is-active" : ""} aria-pressed={selectedQuery === index} onClick={() => { setSelectedQuery(index); if (maskMode === "causal" && selectedKey > index) setSelectedKey(index); }} key={token}><strong>Q{index}</strong><small>{token}</small></button>)}</div>
        </fieldset>
      </div>

      <nav className="attention-phase-rail" aria-label="Attention 计算阶段">
        <button type="button" className="attention-step-arrow" disabled={phaseIndex === 0} onClick={() => setPhaseIndex((index) => Math.max(0, index - 1))} aria-label="上一步"><CaretLeft size={16} /></button>
        <div>{attentionPhases.map((item, index) => <button type="button" className={`${index === phaseIndex ? "is-active" : ""}${index < phaseIndex ? " is-complete" : ""}`} aria-current={index === phaseIndex ? "step" : undefined} onClick={() => setPhaseIndex(index)} key={item.id}><strong>{item.compactLabel}</strong><small>{index + 1}</small></button>)}</div>
        <button type="button" className="attention-step-arrow" disabled={phaseIndex === attentionPhases.length - 1} onClick={() => setPhaseIndex((index) => Math.min(attentionPhases.length - 1, index + 1))} aria-label="下一步"><CaretRight size={16} /></button>
      </nav>

      <div className={`attention-operation-band phase-${phase.id}`}>
        <div>{phase.id === "mix" ? <Lightning size={22} weight="duotone" /> : phase.id === "mask" ? <Eye size={22} weight="duotone" /> : <FunctionIcon size={22} weight="duotone" />}</div>
        <span>当前步骤</span><strong>{phase.label}</strong><code>{phase.formula}</code><p>{phase.explanation}</p>
      </div>

      <div className="attention-main-layout">
        <section className="attention-query-stage" aria-labelledby="attention-query-title">
          <header><div><span>One Query against every Key</span><h3 id="attention-query-title">“{simulation.tokens[selectedQuery]}”同时比较所有允许看到的位置</h3></div><code>Q{selectedQuery} = {formatVector(simulation.q[selectedQuery])}</code></header>

          <div className={`attention-selected-flow reached-${phaseIndex}`}>
            <article className="attention-query-card">
              <span>Query source</span><strong>{simulation.tokens[selectedQuery]}</strong><code>X{selectedQuery} {formatVector(simulation.x[selectedQuery])}</code>
              <i /><code>Q{selectedQuery} {phaseIndex >= 1 ? formatVector(simulation.q[selectedQuery]) : "等待投影"}</code>
            </article>

            <ArrowRight size={22} className="attention-flow-arrow" aria-hidden="true" />

            <div className="attention-key-lanes">
              {simulation.tokens.map((token, keyIndex) => {
                const keyDetail = selectedAttentionDetail(simulation, selectedQuery, keyIndex);
                const masked = phaseIndex >= 4 && keyDetail.masked;
                return (
                  <button type="button" className={`${selectedKey === keyIndex ? "is-selected" : ""}${masked ? " is-masked" : ""}`} aria-pressed={selectedKey === keyIndex} onClick={() => setSelectedKey(keyIndex)} key={token}>
                    <header><span>K{keyIndex}</span><strong>{token}</strong><code>{phaseIndex >= 1 ? formatVector(simulation.k[keyIndex]) : "等待"}</code></header>
                    <div><span>dot</span><strong>{phaseIndex >= 2 ? keyDetail.rawScore : "·"}</strong></div>
                    <div><span>scaled</span><strong>{phaseIndex >= 3 ? formatNumber(keyDetail.scoreAfterScale) : "·"}</strong></div>
                    <div className="attention-lane-weight"><span>{masked ? "mask" : "weight"}</span><strong>{phaseIndex >= 4 && masked ? "0" : phaseIndex >= 5 ? formatPercent(keyDetail.weight) : "·"}</strong><i style={{ width: phaseIndex >= 5 ? `${keyDetail.weight * 100}%` : "0%" }} /></div>
                    <footer><span>V{keyIndex}</span><code>{phaseIndex >= 1 ? formatVector(simulation.v[keyIndex]) : "等待"}</code></footer>
                  </button>
                );
              })}
            </div>

            <ArrowRight size={22} className="attention-flow-arrow" aria-hidden="true" />

            <article className={`attention-output-card${phaseIndex >= 6 ? " is-ready" : ""}`}>
              <span>Output token {selectedQuery}</span><strong>{simulation.tokens[selectedQuery]}′</strong><code>{phaseIndex >= 6 ? formatVector(simulation.output[selectedQuery]) : "等待 P × V"}</code>
              <small>位置不变，向量已混合上下文</small>
            </article>
          </div>

          <div className="attention-role-legend">
            <div><strong>Query</strong><span>我现在想找什么</span></div>
            <div><strong>Key</strong><span>我可以怎样被匹配</span></div>
            <div><strong>Value</strong><span>匹配后真正贡献什么信息</span></div>
          </div>
        </section>

        <aside className="attention-cell-inspector" aria-live="polite">
          <header><span>Selected pair</span><strong>Q{selectedQuery} “{simulation.tokens[selectedQuery]}” → K{selectedKey} “{simulation.tokens[selectedKey]}”</strong></header>
          <div><span>点积</span><code>{phaseIndex >= 2 ? `${simulation.q[selectedQuery].map((value, index) => `${value}×${simulation.k[selectedKey][index]}`).join(" + ")} = ${detail.rawScore}` : "等待 QKᵀ"}</code></div>
          <div><span>缩放</span><code>{phaseIndex >= 3 ? `${detail.rawScore} ÷ ${formatNumber(detail.scaleDivisor)} = ${formatNumber(detail.scoreAfterScale)}` : "等待 Scale"}</code><small>{phaseIndex >= 3 ? scaleMode === "scaled" ? `dₖ=${simulation.headDim}` : "当前关闭缩放，仅用于比较" : "缩放发生在点积之后"}</small></div>
          <div className={phaseIndex >= 4 && detail.masked ? "is-masked" : ""}><span>Mask 判定</span><strong>{phaseIndex < 4 ? "等待 Mask" : detail.masked ? `${selectedKey} > ${selectedQuery}，未来位置不可见` : "允许参与本行 Softmax"}</strong></div>
          <div><span>本 Key 的权重</span><strong>{phaseIndex >= 5 ? formatPercent(detail.weight) : "等待 Softmax"}</strong><code>{phaseIndex >= 5 ? `P[${selectedQuery},${selectedKey}]` : "本行尚未归一化"}</code></div>
          <div><span>对输出的贡献</span><code>{phaseIndex >= 6 ? `${formatPercent(detail.weight)} × ${formatVector(detail.valueVector)}` : "等待 P × V"}</code><strong>{phaseIndex >= 6 ? formatVector(detail.weightedValue) : "尚未混合 Value"}</strong></div>
        </aside>
      </div>

      <section className="attention-matrix-stage" aria-labelledby="attention-matrix-title">
        <header><div><span>Exact T × T state</span><h3 id="attention-matrix-title">行是 Query，列是 Key，Softmax 永远沿一行计算</h3></div><code>{phaseIndex < 2 ? "score matrix 尚未产生" : phase.formula}</code></header>
        <div className="attention-matrix-layout">
          <div className="attention-matrix-scroll">
            <div className="attention-score-matrix">
              <span className="attention-matrix-corner">Q \ K</span>
              {simulation.tokens.map((token, index) => <button type="button" className={selectedKey === index ? "is-selected" : ""} onClick={() => setSelectedKey(index)} key={`h-${token}`}><strong>K{index}</strong><small>{token}</small></button>)}
              {simulation.tokens.flatMap((token, queryIndex) => [
                <button type="button" className={`attention-row-header${selectedQuery === queryIndex ? " is-selected" : ""}`} onClick={() => setSelectedQuery(queryIndex)} key={`r-${token}`}><strong>Q{queryIndex}</strong><small>{token}</small></button>,
                ...simulation.tokens.map((_, keyIndex) => {
                  const value = matrix?.[queryIndex]?.[keyIndex] ?? null;
                  const masked = phaseIndex >= 4 && simulation.maskedScores[queryIndex][keyIndex] === null;
                  const strength = phaseIndex >= 5 ? simulation.weights[queryIndex][keyIndex] : phaseIndex >= 2 ? simulation.rawScores[queryIndex][keyIndex] / 3 : 0;
                  return <button type="button" className={`${selectedQuery === queryIndex && selectedKey === keyIndex ? "is-selected" : ""}${masked ? " is-masked" : ""}`} aria-label={`Q${queryIndex} 到 K${keyIndex}`} onClick={() => { setSelectedQuery(queryIndex); setSelectedKey(keyIndex); }} key={`${queryIndex}-${keyIndex}`}><i style={{ height: `${Math.max(0, Math.min(1, strength)) * 100}%` }} /><strong>{matrixText(value, phaseIndex)}</strong><small>{masked ? "future" : `q${queryIndex}·k${keyIndex}`}</small></button>;
                }),
              ])}
            </div>
          </div>

          <aside>
            <span>当前第 {selectedQuery} 行</span><strong>{phaseIndex >= 5 ? `Σ weights = ${formatNumber(simulation.weights[selectedQuery].reduce((sum, value) => sum + value, 0))}` : `${phase.compactLabel} row`}</strong>
            <div>{simulation.weights[selectedQuery].map((weight, keyIndex) => {
              const currentValue = matrix?.[selectedQuery]?.[keyIndex] ?? null;
              return <span className={keyIndex === selectedKey ? "is-selected" : ""} key={keyIndex}><i style={{ width: phaseIndex >= 5 ? `${weight * 100}%` : "0%" }} /><small>K{keyIndex}</small><strong>{phaseIndex >= 5 ? formatPercent(weight) : matrixText(currentValue, phaseIndex)}</strong></span>;
            })}</div>
            <p>{maskMode === "causal" ? `Q${selectedQuery} 只有 K0..K${selectedQuery} 可以分配概率。被 Mask 的位置在 Softmax 后严格为 0。` : `双向模式下 Q${selectedQuery} 可以向所有 ${simulation.tokens.length} 个 Key 分配概率。`}</p>
          </aside>
        </div>
      </section>

      <section className="attention-value-mix" aria-labelledby="attention-mix-title">
        <header><div><span>Weighted Value reduction</span><h3 id="attention-mix-title">概率只是路由系数，最终被聚合的是 Value</h3></div><code>O{selectedQuery} = Σⱼ P[{selectedQuery},j]Vⱼ</code></header>
        <div>
          {simulation.tokens.map((token, keyIndex) => {
            const weight = simulation.weights[selectedQuery][keyIndex];
            const contribution = simulation.v[keyIndex].map((value) => value * weight);
            return <article className={`${keyIndex === selectedKey ? "is-selected" : ""}${phaseIndex >= 5 && weight === 0 ? " is-zero" : ""}`} key={token}><header><span>{phaseIndex >= 5 ? formatPercent(weight) : "等待 P"}</span><strong>× V{keyIndex} {token}</strong></header><code>{phaseIndex >= 1 ? formatVector(simulation.v[keyIndex]) : "等待 V 投影"}</code><ArrowRight size={14} aria-hidden="true" /><strong>{phaseIndex >= 6 ? formatVector(contribution) : "等待 PV"}</strong><i><span style={{ width: phaseIndex >= 5 ? `${weight * 100}%` : "0%" }} /></i></article>;
          })}
          <i className="attention-mix-equals">Σ</i>
          <article className="attention-mix-result"><header><span>Output</span><strong>O{selectedQuery}</strong></header><code>{phaseIndex >= 6 ? formatVector(simulation.output[selectedQuery]) : "等待所有 Value contribution"}</code><small>继续送入输出投影与残差路径</small></article>
        </div>
      </section>

      <section className="attention-projection-stage" aria-labelledby="attention-projection-title">
        <header><div><span>Same X, three learned roles</span><h3 id="attention-projection-title">Q、K、V 不是三份原始输入，而是三组 Linear 投影结果</h3></div><code>本页 Wq 取单位矩阵，方便手算</code></header>
        <div className="attention-projection-table">
          <span className="attention-projection-corner">token</span><strong>X</strong><strong className="role-q">Q = XWq</strong><strong className="role-k">K = XWk</strong><strong className="role-v">V = XWv</strong>
          {simulation.tokens.flatMap((token, index) => [
            <button type="button" className={selectedQuery === index ? "is-selected" : ""} onClick={() => setSelectedQuery(index)} key={`t-${token}`}><strong>{token}</strong><small>position {index}</small></button>,
            <code key={`x-${token}`}>{formatVector(simulation.x[index])}</code>,
            <code className="role-q" key={`q-${token}`}>{formatVector(simulation.q[index])}</code>,
            <code className="role-k" key={`k-${token}`}>{formatVector(simulation.k[index])}</code>,
            <code className="role-v" key={`v-${token}`}>{formatVector(simulation.v[index])}</code>,
          ])}
        </div>
        <div className="attention-projection-weights">
          <div><span>Wq</span><code>{projectionLabel(ATTENTION_WQ)}</code><small>保留三个教学特征</small></div>
          <div><span>Wk</span><code>{projectionLabel(ATTENTION_WK)}</code><small>交换后两个特征，产生不同匹配空间</small></div>
          <div><span>Wv</span><code>{projectionLabel(ATTENTION_WV)}</code><small>把三维信息映射为二维输出内容</small></div>
        </div>
      </section>

      <section className="attention-head-stage" aria-labelledby="attention-head-title">
        <header><div><span>Multi-head is reshape plus parallel attention</span><h3 id="attention-head-title">多头不是把 token 分组，而是把投影后的特征维拆开</h3></div><div className="attention-head-controls">{[1, 2, 4].map((count) => <button type="button" className={headCount === count ? "is-active" : ""} aria-pressed={headCount === count} onClick={() => setHeadCount(count)} key={count}>{count} heads</button>)}</div></header>
        <div className="attention-head-flow">
          <article><span>Linear projection</span><strong>[T, D]</strong><code>[4, 8]</code><small>一次大矩阵乘法可同时产生所有 head</small></article>
          <ArrowRight size={18} aria-hidden="true" />
          <article><span>Reshape</span><strong>[h, T, dₕ]</strong><code>[{headLayout.headCount}, 4, {headLayout.headDim}]</code><small>D = h × dₕ = {headLayout.headCount} × {headLayout.headDim}</small></article>
          <ArrowRight size={18} aria-hidden="true" />
          <div className="attention-heads">{Array.from({ length: headCount }, (_, head) => <span className={`head-tone-${head % 4}`} key={head}><strong>Head {head}</strong><code>4 × 4 scores</code><small>独立 Softmax</small></span>)}</div>
          <ArrowRight size={18} aria-hidden="true" />
          <article><span>Concat + Wo</span><strong>[T, D]</strong><code>[4, 8]</code><small>拼回模型维度，再做输出投影</small></article>
        </div>
        <p><strong>关键点：</strong>每个 head 有自己的投影子空间，因此可以学习不同关系；但所有 head 的 score matrix 都是 T × T。</p>
      </section>

      <section className="attention-cost-stage" aria-labelledby="attention-cost-title">
        <header><div><span>Why long context is expensive</span><h3 id="attention-cost-title">QKV 随 T 线性增长，Score 元素随 T² 增长</h3><p>下面固定 32 heads、head_dim=128、FP16。Score bytes 表示朴素实现将完整 score tensor 物化时的一份逻辑大小，不代表 fused kernel 一定写入同样多的 HBM。</p></div><code>score shape = [32, T, T]</code></header>
        <div className="attention-cost-cards">
          {costExamples.map((cost) => <article key={cost.sequenceLength}><header><span>Sequence length</span><strong>T = {cost.sequenceLength.toLocaleString()}</strong></header><div><span>Score elements</span><strong>{formatCount(cost.scoreElements)}</strong></div><div><span>Score FP16</span><strong>{formatBytes(cost.scoreBytes)}</strong></div><div><span>QKV FP16</span><strong>{formatBytes(cost.qkvBytes)}</strong></div><div><span>QKᵀ + PV</span><strong>{formatCount(cost.totalAttentionFlops)} FLOPs</strong></div><i><span style={{ width: `${Math.max(2, cost.scoreBytes / maxScoreBytes * 100)}%` }} /></i></article>)}
        </div>
        <div className="attention-growth-proof"><div><span>T 翻 2 倍</span><strong>QKV × 2</strong><code>O(TD)</code></div><div><span>T 翻 2 倍</span><strong>Score × 4</strong><code>O(T²)</code></div><p>这就是 FlashAttention 要解决的系统问题：数学结果保持完全一致，但用分块和在线 Softmax 避免把完整 T×T 中间矩阵反复写入 HBM。</p></div>
      </section>

      <section className="attention-sources" aria-labelledby="attention-sources-title">
        <header><span>Primary references</span><h3 id="attention-sources-title">公式与系统连接依据</h3></header>
        <div>
          <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer"><strong>Attention Is All You Need</strong><span>Scaled Dot-Product Attention 与 Multi-Head Attention 原始定义</span></a>
          <a href="https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html" target="_blank" rel="noreferrer"><strong>PyTorch SDPA</strong><span>scale、attention bias、causal mask 与 Softmax 的参考实现语义</span></a>
          <a href="https://arxiv.org/abs/2205.14135" target="_blank" rel="noreferrer"><strong>FlashAttention</strong><span>Attention 的 T² 中间状态如何连接到 HBM 与片上 SRAM</span></a>
        </div>
      </section>
    </section>
  );
}
