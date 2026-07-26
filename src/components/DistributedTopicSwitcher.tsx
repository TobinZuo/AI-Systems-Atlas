import { ArrowRight } from "@phosphor-icons/react/ArrowRight";

export type DistributedTopicId = "process-rank" | "ddp" | "zero-1" | "fsdp" | "compare";

const topics: Array<{
  id: DistributedTopicId;
  label: string;
  summary: string;
  route: string;
}> = [
  { id: "process-rank", label: "Process / Rank", summary: "建立任务身份与通信组", route: "/distributed/process-rank" },
  { id: "ddp", label: "DDP", summary: "复制模型，同步梯度", route: "/distributed/ddp" },
  { id: "zero-1", label: "ZeRO-1", summary: "分片 optimizer state", route: "/distributed/zero-1" },
  { id: "fsdp", label: "FSDP", summary: "分片参数、梯度与状态", route: "/distributed/fsdp" },
  { id: "compare", label: "横向对比", summary: "同一口径看状态与通信", route: "/distributed/compare" },
];

export function DistributedTopicSwitcher({ current }: { current: DistributedTopicId }) {
  return (
    <nav className="distributed-topic-switcher" aria-label="分布式训练专题">
      <span className="switcher-label">分布式训练主线</span>
      <div>
        {topics.map((topic, index) => (
          <span className="switcher-topic-wrap" key={topic.id}>
            <a
              href={`#${topic.route}`}
              className={topic.id === current ? "is-current" : ""}
              aria-current={topic.id === current ? "page" : undefined}
            >
              <strong>{topic.label}</strong>
              <small>{topic.summary}</small>
            </a>
            {index < topics.length - 1 && <ArrowRight size={15} aria-hidden="true" />}
          </span>
        ))}
      </div>
    </nav>
  );
}
