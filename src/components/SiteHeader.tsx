import { Cube } from "@phosphor-icons/react/Cube";
import { GithubLogo } from "@phosphor-icons/react/GithubLogo";
import { Moon } from "@phosphor-icons/react/Moon";
import { Sun } from "@phosphor-icons/react/Sun";

type Theme = "light" | "dark";

interface SiteHeaderProps {
  theme: Theme;
  current: "atlas" | "paths" | "ddp" | "zero-1" | "fsdp" | "compare" | "other";
  onToggleTheme: () => void;
}

export function SiteHeader({ theme, current, onToggleTheme }: SiteHeaderProps) {
  return (
    <header className="site-header">
      <a className="brand" href="#/" aria-label="返回 AI Systems Atlas 知识地图">
        <span className="brand-mark" aria-hidden="true">
          <Cube size={22} weight="duotone" />
        </span>
        <span>AI Systems Atlas</span>
      </a>

      <nav aria-label="主导航">
        <a href="#/" aria-current={current === "atlas" ? "page" : undefined}>
          知识地图
        </a>
        <a href="#/paths" aria-current={current === "paths" ? "page" : undefined}>
          学习路线
        </a>
        <a href="#/distributed/ddp" aria-current={current === "ddp" ? "page" : undefined}>DDP</a>
        <a href="#/distributed/zero-1" aria-current={current === "zero-1" ? "page" : undefined}>ZeRO-1</a>
        <a href="#/distributed/fsdp" aria-current={current === "fsdp" ? "page" : undefined}>FSDP</a>
        <a href="#/distributed/compare" aria-current={current === "compare" ? "page" : undefined}>横向对比</a>
        <a
          className="github-link"
          href="https://github.com/TobinZuo/AI-Systems-Atlas"
          target="_blank"
          rel="noreferrer"
        >
          <GithubLogo size={18} weight="fill" aria-hidden="true" />
          <span>GitHub</span>
        </a>
        <button
          type="button"
          className="theme-toggle"
          onClick={onToggleTheme}
          aria-label={`切换到${theme === "dark" ? "浅色" : "深色"}主题`}
        >
          {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </nav>
    </header>
  );
}
