import { Component, type ErrorInfo, type ReactNode } from "react";
import { WarningCircle } from "@phosphor-icons/react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("3D scene failed to render", error, info);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="scene-state scene-state--error" role="alert">
          <WarningCircle size={28} weight="duotone" aria-hidden="true" />
          <strong>The 3D view could not start.</strong>
          <span>
            You can still use the timeline and component inspector below.
          </span>
          <button
            type="button"
            className="text-button"
            onClick={() => this.setState({ hasError: false })}
          >
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
