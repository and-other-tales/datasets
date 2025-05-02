import React from "react";
import { MessageContainer } from "./shared";

interface GenericInterruptMessageProps {
  message: {
    id: string;
    content: string;
  };
}

export function GenericInterruptMessage({ message }: GenericInterruptMessageProps) {
  return (
    <MessageContainer className="bg-destructive/10 border border-destructive/20">
      <div className="flex items-center gap-2 text-destructive">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="lucide lucide-x"
        >
          <path d="M18 6 6 18" />
          <path d="m6 6 12 12" />
        </svg>
        <span>{message.content}</span>
      </div>
    </MessageContainer>
  );
}