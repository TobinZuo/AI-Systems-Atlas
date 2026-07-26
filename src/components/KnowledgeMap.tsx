import { ArrowRight } from "@phosphor-icons/react/ArrowRight";
import { Atom } from "@phosphor-icons/react/Atom";
import { Circuitry } from "@phosphor-icons/react/Circuitry";
import { Cpu } from "@phosphor-icons/react/Cpu";
import { Database } from "@phosphor-icons/react/Database";
import { Graph } from "@phosphor-icons/react/Graph";
import { Lightning } from "@phosphor-icons/react/Lightning";
import { Network } from "@phosphor-icons/react/Network";
import { useState } from "react";
import {
  knowledgeDomains,
  topicIndex,
  type DomainIcon,
  type TopicStatus,
} from "../content/topics";

const domainIcons = {
  database: Database,
  circuitry: Circuitry,
  graph: Graph,
  cpu: Cpu,
  network: Network,
  lightning: Lightning,
  atom: Atom,
} satisfies Record<DomainIcon, typeof Database>;

const statusLabels: Record<TopicStatus, string> = {
  available: "可交互",
  next: "优先制作",
  mapped: "已入地图",
};

export function KnowledgeMap() {
  const [selectedTopicId, setSelectedTopicId] = useState("ddp");
  const selectedTopic = topicIndex.get(selectedTopicId) ?? topicIndex.get("ddp")!;
  const selectedDomain = knowledgeDomains.find((domain) => domain.id === selectedTopic.domainId)!;
  const prerequisiteIds = new Set(selectedTopic.prerequisites ?? []);
  const prerequisites = [...prerequisiteIds]
    .map((topicId) => topicIndex.get(topicId))
    .filter((topic) => topic !== undefined);

  return (
    <section className="knowledge-atlas" id="knowledge-atlas" aria-labelledby="knowledge-atlas-title">
      <div className="atlas-section-heading">
        <div>
          <p className="eyebrow">Knowledge map</p>
          <h2 id="knowledge-atlas-title">先看全貌，再沿依赖关系下钻。</h2>
        </div>
        <p>
          每一行是一个知识域。点选主题可以查看它解决的问题、前置概念和当前制作状态。
        </p>
      </div>

      <div className="knowledge-layout">
        <div className="domain-map" aria-label="大模型知识主题">
          {knowledgeDomains.map((domain) => {
            const Icon = domainIcons[domain.icon];
            return (
              <article className="domain-row" data-domain={domain.id} key={domain.id}>
                <header className="domain-heading">
                  <span className="domain-index">{domain.index}</span>
                  <span className="domain-icon" aria-hidden="true">
                    <Icon size={21} weight="duotone" />
                  </span>
                  <div>
                    <h3>{domain.title}</h3>
                    <p>{domain.question}</p>
                  </div>
                </header>

                <div className="topic-cloud">
                  {domain.topics.map((topic) => {
                    const isSelected = topic.id === selectedTopic.id;
                    const isPrerequisite = prerequisiteIds.has(topic.id);
                    return (
                      <button
                        type="button"
                        className={`topic-node${isSelected ? " is-selected" : ""}${isPrerequisite ? " is-prerequisite" : ""}`}
                        aria-pressed={isSelected}
                        onClick={() => setSelectedTopicId(topic.id)}
                        key={topic.id}
                      >
                        <span>{topic.title}</span>
                        <i className={`topic-status-dot status-${topic.status}`} aria-hidden="true" />
                        <span className="visually-hidden">，{statusLabels[topic.status]}</span>
                      </button>
                    );
                  })}
                </div>
              </article>
            );
          })}
        </div>

        <aside className="topic-inspector" data-domain={selectedDomain.id} aria-live="polite">
          <div className="inspector-kicker">
            <span>{selectedDomain.index}</span>
            <strong>{selectedDomain.title}</strong>
          </div>
          <div className="inspector-title-row">
            <h3>{selectedTopic.title}</h3>
            <span className={`topic-status status-${selectedTopic.status}`}>
              {statusLabels[selectedTopic.status]}
            </span>
          </div>
          <p className="inspector-description">{selectedTopic.description}</p>

          <div className="inspector-block">
            <span>它属于哪一层</span>
            <strong>{selectedDomain.description}</strong>
          </div>

          <div className="inspector-block">
            <span>先理解这些</span>
            {prerequisites.length > 0 ? (
              <div className="prerequisite-list">
                {prerequisites.map((topic) => (
                  <button type="button" onClick={() => setSelectedTopicId(topic.id)} key={topic.id}>
                    {topic.title}
                  </button>
                ))}
              </div>
            ) : (
              <strong>可以从这里开始</strong>
            )}
          </div>

          {selectedTopic.route ? (
            <a className="inspector-action" href={`#${selectedTopic.route}`}>
              进入完整专题
              <ArrowRight size={17} aria-hidden="true" />
            </a>
          ) : (
            <p className="inspector-note">
              已纳入知识地图。后续会补齐可操作模型、系统时间线与源码证据。
            </p>
          )}
        </aside>
      </div>

      <div className="map-legend" aria-label="主题状态图例">
        <span><i className="topic-status-dot status-available" />可交互专题</span>
        <span><i className="topic-status-dot status-next" />下一批制作</span>
        <span><i className="topic-status-dot status-mapped" />已进入整体地图</span>
        <span className="legend-hint">被高亮的描边节点是当前主题的前置概念</span>
      </div>
    </section>
  );
}
