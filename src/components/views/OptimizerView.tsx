import { ArrowRight } from "@phosphor-icons/react";
import type { SimulationEvent } from "../../domain/simulation";

export function OptimizerView({ event }: { event: SimulationEvent }) {
  const active = event.id === "optimizer-update";
  return (
    <div className="optimizer-view detail-view">
      <div className="view-intro-row">
        <div><span className="view-kicker">AdamW · element 0</span><h3>拿平均梯度 277.75 走完一次参数更新</h3></div>
        <div className="formula-chip"><code>lr=0.001 · β₁=0.9 · β₂=0.999 · wd=0.01</code><span>初始 θ=1, m=0, v=0</span></div>
      </div>
      <div className={`optimizer-pipeline${active ? " is-live" : ""}`}>
        <div><small>输入状态 · HBM</small><strong>θ = 1.00000</strong><code>g = 277.75<br/>m = 0<br/>v = 0</code></div>
        <ArrowRight size={19} />
        <div><small>一阶矩</small><strong>m₁ = 27.775</strong><code>0.9m + 0.1g</code></div>
        <ArrowRight size={19} />
        <div><small>二阶矩</small><strong>v₁ ≈ 77.145</strong><code>0.999v + 0.001g²</code></div>
        <ArrowRight size={19} />
        <div><small>Bias correction</small><strong>m̂=277.75</strong><code>v̂≈77145.06<br/>m̂/√v̂≈1</code></div>
        <ArrowRight size={19} />
        <div className="optimizer-result"><small>AdamW 输出</small><strong>θ₁ ≈ 0.99899</strong><code>θ(1−lr·wd) − lr·m̂/(√v̂+ε)</code></div>
      </div>
      <div className="adamw-contrast">
        <section><span>Adam + L2</span><strong>先把 λθ 加入梯度</strong><p>这个值会进入 m 和 v 的统计，正则强度受自适应缩放影响。</p></section>
        <section className="is-accent"><span>AdamW</span><strong>Weight decay 与梯度更新解耦</strong><p>先按 Adam 方向更新，再直接衰减参数；λθ 不污染动量估计。</p></section>
        <section><span>为什么 4 个 rank 仍一致</span><strong>相同 θ、g、m、v → 相同 θ₁</strong><p>每张 GPU 独立运行 kernel，不需要再次广播参数。</p></section>
      </div>
    </div>
  );
}
