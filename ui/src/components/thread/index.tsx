"use client";

import React, { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useStream } from "@/providers/Stream";
import { AIMessage } from "./messages/ai";
import { HumanMessage } from "./messages/human";
import { ToolCallMessage } from "./messages/tool-calls";
import { GenericInterruptMessage } from "./messages/generic-interrupt";
import useStickToBottom from "use-stick-to-bottom";

interface ThreadProps {
  showHistory?: boolean;
}

export function Thread({ showHistory = false }: ThreadProps) {
  const { messages, sendMessage, isLoading, interrupt } = useStream();
  const [messageInput, setMessageInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  
  const { stickToBottom } = useStickToBottom(scrollRef);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!messageInput.trim()) return;
    
    const messageCopy = messageInput;
    setMessageInput("");
    await sendMessage(messageCopy);
    stickToBottom();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-1 overflow-hidden flex">
        {showHistory && (
          <div className="w-64 border-r border-border p-4 hidden md:block">
            <h2 className="text-lg font-medium mb-4">History</h2>
            {/* History could be implemented here if needed */}
          </div>
        )}
        
        <div className="flex-1 flex flex-col h-full overflow-hidden">
          <div 
            ref={scrollRef}
            className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin"
          >
            {messages.length === 0 && (
              <div className="text-center text-muted-foreground p-8">
                <p className="mb-2 text-xl">Welcome to the OtherTales Datasets Agent</p>
                <p>Send a message to get started with dataset creation.</p>
              </div>
            )}
            
            {messages.map((message) => {
              if (message.type === "human") {
                return (
                  <HumanMessage
                    key={message.id}
                    message={{
                      id: message.id,
                      content: String(message.content),
                      metadata: message.metadata,
                    }}
                  />
                );
              } else if (message.type === "ai") {
                return (
                  <AIMessage
                    key={message.id}
                    message={{
                      id: message.id,
                      content: String(message.content),
                      metadata: message.metadata,
                    }}
                  />
                );
              } else if (message.type === "tool" || message.type === "tool_result") {
                return (
                  <ToolCallMessage
                    key={message.id}
                    message={{
                      id: message.id,
                      type: message.type,
                      content: message.content,
                      metadata: message.metadata,
                    }}
                  />
                );
              } else if (message.type === "interrupt") {
                return (
                  <GenericInterruptMessage
                    key={message.id}
                    message={{
                      id: message.id,
                      content: String(message.content),
                    }}
                  />
                );
              }
              return null;
            })}
            
            {isLoading && (
              <div className="flex items-center justify-center p-4">
                <div className="animate-pulse">Thinking...</div>
              </div>
            )}
          </div>
          
          <div className="border-t border-border p-4">
            <form onSubmit={handleSendMessage} className="flex gap-2">
              <Textarea
                value={messageInput}
                onChange={(e) => setMessageInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type your message..."
                className="flex-1 resize-none"
                disabled={isLoading}
              />
              
              {isLoading ? (
                <Button 
                  type="button" 
                  onClick={() => interrupt()}
                  variant="destructive"
                >
                  Stop
                </Button>
              ) : (
                <Button type="submit" disabled={!messageInput.trim()}>
                  Send
                </Button>
              )}
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}