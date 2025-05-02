import React from "react";
import { MessageContainer, MessageMetadata } from "./shared";

interface HumanMessageProps {
  message: MessageMetadata;
}

export function HumanMessage({ message }: HumanMessageProps) {
  return (
    <MessageContainer className="bg-secondary border border-border">
      <div className="flex items-start gap-3">
        <div className="rounded-full bg-primary text-primary-foreground w-8 h-8 flex items-center justify-center text-sm font-medium">
          Y
        </div>
        <div className="flex-1">
          <div className="font-medium mb-1">You</div>
          <div className="text-foreground whitespace-pre-wrap">{message.content}</div>
        </div>
      </div>
    </MessageContainer>
  );
}