import React from "react";
import { MessageContainer } from "./shared";

interface ToolCallMessageProps {
  message: {
    id: string;
    type: string;
    content: unknown;
    metadata?: Record<string, unknown>;
  };
}

export function ToolCallMessage({ message }: ToolCallMessageProps) {
  const isToolCall = message.type === "tool";
  const isToolResult = message.type === "tool_result";
  
  let displayContent = "";
  let toolName = "Tool";
  
  if (isToolCall && typeof message.content === "object" && message.content !== null) {
    const content = message.content as any;
    toolName = content.name || "Tool";
    try {
      displayContent = JSON.stringify(content.args || content, null, 2);
    } catch (error) {
      displayContent = "Error parsing tool call content";
    }
  } else if (isToolResult) {
    try {
      displayContent = typeof message.content === "string" 
        ? message.content
        : JSON.stringify(message.content, null, 2);
    } catch (error) {
      displayContent = "Error parsing tool result content";
    }
  }

  return (
    <MessageContainer className={isToolCall ? "bg-muted" : "bg-muted/50"}>
      <div className="flex items-start gap-3">
        <div 
          className={`rounded-full ${
            isToolCall ? "bg-secondary" : "bg-secondary/80"
          } text-foreground w-8 h-8 flex items-center justify-center text-sm font-medium`}
        >
          {isToolCall ? "T" : "R"}
        </div>
        <div className="flex-1">
          <div className="font-medium mb-1">
            {isToolCall ? `Tool: ${toolName}` : "Tool Result"}
          </div>
          <pre className="bg-muted-foreground/5 p-2 rounded overflow-x-auto text-xs">
            <code>{displayContent}</code>
          </pre>
        </div>
      </div>
    </MessageContainer>
  );
}