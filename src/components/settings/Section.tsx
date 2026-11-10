import type { ReactNode } from "react";

export function Section({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <section>
      <h2 className="mb-2 text-xs font-medium uppercase tracking-wide text-muted">
        {title}
      </h2>
      <div className="flex flex-col">{children}</div>
    </section>
  );
}
