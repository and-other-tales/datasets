import React from "react";

export interface MessageMetadata {
  id: string;
  content: string;
  metadata?: Record<string, unknown>;
}

interface MessageContainerProps {
  children: React.ReactNode;
  className?: string;
}

export function MessageContainer({
  children,
  className = "",
}: MessageContainerProps) {
  return (
    <div
      className={`p-4 rounded-lg animate-in fade-in slide-in-from-bottom-2 duration-300 ${className}`}
    >
      {children}
    </div>
  );
}