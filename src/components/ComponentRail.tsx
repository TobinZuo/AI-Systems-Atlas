import {
  ArrowsLeftRight,
  Brain,
  Cpu,
  Cube,
  Database,
  Function,
} from "@phosphor-icons/react";
import type { ComponentId } from "../domain/simulation";
import { useSimulationStore } from "../store/simulation";

const groups: Array<{
  label: string;
  items: Array<{
    id: ComponentId;
    name: string;
    icon: typeof Cpu;
  }>;
}> = [
  {
    label: "Training",
    items: [
      { id: "python", name: "Rank process", icon: Function },
      { id: "autograd", name: "Autograd", icon: Brain },
      { id: "ddp", name: "DDP reducer", icon: Cube },
    ],
  },
  {
    label: "Hardware",
    items: [
      { id: "cpu", name: "Host CPU", icon: Cpu },
      { id: "gpu-0", name: "GPU ranks", icon: Cube },
      { id: "sm", name: "SM and warps", icon: Cpu },
      { id: "hbm", name: "GPU HBM", icon: Database },
    ],
  },
  {
    label: "Communication",
    items: [
      { id: "nccl", name: "NCCL", icon: ArrowsLeftRight },
      { id: "nvlink", name: "NVLink ring", icon: ArrowsLeftRight },
      { id: "optimizer", name: "Optimizer", icon: Function },
    ],
  },
];

export function ComponentRail() {
  const selected = useSimulationStore((state) => state.selectedComponent);
  const select = useSimulationStore((state) => state.selectComponent);

  return (
    <nav className="component-rail" aria-label="System components">
      {groups.map((group) => (
        <div className="rail-group" key={group.label}>
          <p>{group.label}</p>
          {group.items.map((item) => {
            const Icon = item.icon;
            const active =
              selected === item.id ||
              (item.id === "gpu-0" && selected.startsWith("gpu-"));
            return (
              <button
                type="button"
                className={active ? "rail-item is-active" : "rail-item"}
                key={item.id}
                onClick={() => select(item.id)}
                aria-pressed={active}
              >
                <Icon size={17} weight={active ? "fill" : "regular"} />
                <span>{item.name}</span>
              </button>
            );
          })}
        </div>
      ))}
    </nav>
  );
}
