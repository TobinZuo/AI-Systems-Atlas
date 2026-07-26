import {
  Atom,
  BracketsCurly,
  ChartLineUp,
  CirclesFour,
  Graph,
  Stack,
} from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";

const paths = [
  {
    title: "Distributed training",
    description:
      "Trace gradients through DDP, collectives, GPU memory, and interconnects.",
    status: "Interactive now",
    icon: Graph,
    featured: true,
  },
  {
    title: "Tensor foundations",
    description: "See strides, broadcasting, einsum, and automatic differentiation.",
    status: "Coming next",
    icon: BracketsCurly,
  },
  {
    title: "Transformer anatomy",
    description: "Open attention, MLP, residual, and normalization paths.",
    status: "Planned",
    icon: Stack,
  },
  {
    title: "Training mechanics",
    description: "Inspect optimizers, mixed precision, clipping, and checkpoints.",
    status: "Planned",
    icon: CirclesFour,
  },
  {
    title: "Scaling behavior",
    description: "Manipulate compute, data, parameters, and performance tradeoffs.",
    status: "Planned",
    icon: ChartLineUp,
  },
  {
    title: "Generative models",
    description: "Explore diffusion, multimodal systems, and generation pipelines.",
    status: "Planned",
    icon: Atom,
  },
];

export function LearningPaths() {
  const reducedMotion = useReducedMotion();

  return (
    <section className="learning-paths" id="learning-paths">
      <div className="section-copy">
        <h2>One atlas, multiple scales</h2>
        <p>
          Start with a concept, then descend through framework, runtime, hardware,
          memory, and network behavior.
        </p>
      </div>

      <div className="path-grid">
        {paths.map((path, index) => {
          const Icon = path.icon;
          return (
            <motion.article
              className={path.featured ? "path-card is-featured" : "path-card"}
              key={path.title}
              initial={reducedMotion ? false : { opacity: 0, y: 18 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.25 }}
              transition={{
                duration: 0.45,
                delay: reducedMotion ? 0 : index * 0.045,
                ease: [0.16, 1, 0.3, 1],
              }}
            >
              <div className="path-icon" aria-hidden="true">
                <Icon size={24} weight="duotone" />
              </div>
              <div>
                <h3>{path.title}</h3>
                <p>{path.description}</p>
              </div>
              <span className="path-status">{path.status}</span>
            </motion.article>
          );
        })}
      </div>
    </section>
  );
}
