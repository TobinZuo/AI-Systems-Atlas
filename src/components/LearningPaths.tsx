import { Atom } from "@phosphor-icons/react/Atom";
import { BracketsCurly } from "@phosphor-icons/react/BracketsCurly";
import { ChartLineUp } from "@phosphor-icons/react/ChartLineUp";
import { CirclesFour } from "@phosphor-icons/react/CirclesFour";
import { Graph } from "@phosphor-icons/react/Graph";
import { Stack } from "@phosphor-icons/react/Stack";

const paths = [
  {
    title: "分布式训练",
    description:
      "沿着梯度检查 DDP、集合通信、GPU 显存与互连。",
    status: "当前可交互",
    icon: Graph,
    featured: true,
  },
  {
    title: "Tensor 基础",
    description: "观察 stride、broadcast、einsum 与自动微分。",
    status: "后续实现",
    icon: BracketsCurly,
  },
  {
    title: "Transformer 剖面",
    description: "拆开 attention、MLP、残差与归一化路径。",
    status: "规划中",
    icon: Stack,
  },
  {
    title: "训练机制",
    description: "检查 optimizer、混合精度、梯度裁剪与 checkpoint。",
    status: "规划中",
    icon: CirclesFour,
  },
  {
    title: "规模化规律",
    description: "调整计算量、数据量和参数量，观察性能权衡。",
    status: "规划中",
    icon: ChartLineUp,
  },
  {
    title: "生成模型",
    description: "探索 diffusion、多模态系统与生成流水线。",
    status: "规划中",
    icon: Atom,
  },
];

export function LearningPaths() {
  return (
    <section className="learning-paths" id="learning-paths">
      <div className="section-copy">
        <h2>一张 Atlas，覆盖多个系统尺度</h2>
        <p>
          从一个概念开始，依次下钻到框架、运行时、硬件、内存和网络行为。
        </p>
      </div>

      <div className="path-grid">
        {paths.map((path, index) => {
          const Icon = path.icon;
          return (
            <article
              className={path.featured ? "path-card is-featured" : "path-card"}
              key={path.title}
            >
              <div className="path-icon" aria-hidden="true">
                <Icon size={24} weight="duotone" />
              </div>
              <div>
                <h3>{path.title}</h3>
                <p>{path.description}</p>
              </div>
              <span className="path-status">{path.status}</span>
            </article>
          );
        })}
      </div>
    </section>
  );
}
