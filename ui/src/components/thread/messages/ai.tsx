import React from "react";
import { MessageContainer, MessageMetadata } from "./shared";
import { MarkdownText } from "../markdown-text";

interface AIMessageProps {
  message: MessageMetadata;
}

export function AIMessage({ message }: AIMessageProps) {
  return (
    <MessageContainer className="bg-primary/5">
      <div className="flex items-start gap-3">
        <div className="rounded-full bg-primary text-primary-foreground w-8 h-8 flex items-center justify-center text-sm font-medium">
          A
        </div>
        <div className="flex-1">
          <div className="font-medium mb-1">Assistant</div>
          <MarkdownText content={message.content} />
        </div>
      </div>
    </MessageContainer>
  );
}