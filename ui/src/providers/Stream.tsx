"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { useThread } from "./Thread";
import { LangGraphClient } from "./client";
import { toast } from "sonner";

export type Message = {
  id: string;
  type: "human" | "ai" | "tool" | "tool_result" | "system" | "interrupt";
  content: string | unknown;
  metadata?: Record<string, unknown>;
};

interface StreamContextType {
  messages: Message[];
  sendMessage: (message: string) => Promise<void>;
  isLoading: boolean;
  error: Error | null;
  interrupt: () => Promise<void>;
  client: LangGraphClient;
}

const StreamContext = createContext<StreamContextType | undefined>(undefined);

export function StreamProvider({ 
  children, 
  apiUrl = "/",
  assistantId
}: { 
  children: React.ReactNode;
  apiUrl: string | null;
  assistantId: string | null;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const { activeThread, setActiveThread } = useThread();
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const [client] = useState(() => new LangGraphClient(apiUrl || "/"));

  // Function to fetch messages for the current thread
  const fetchMessages = async () => {
    if (!activeThread || !assistantId) return;
    
    try {
      const messagesData = await client.getMessages(assistantId, activeThread.id);
      const formattedMessages = messagesData.map((message) => ({
        id: message.id,
        type: message.type === "human" 
          ? "human" 
          : message.type === "ai" 
            ? "ai" 
            : message.type === "tool" 
              ? "tool" 
              : message.type === "tool_result" 
                ? "tool_result" 
                : "system",
        content: message.content,
        metadata: message.metadata,
      }));
      
      setMessages(formattedMessages);
    } catch (err) {
      console.error("Error fetching messages:", err);
      setError(err instanceof Error ? err : new Error("Failed to fetch messages"));
    }
  };

  // Initialize thread if needed and connect to EventSource
  useEffect(() => {
    const initializeThread = async () => {
      if (!activeThread || !assistantId) return;
      
      try {
        // Create thread on the server if needed
        try {
          await client.getThread(assistantId, activeThread.id);
        } catch (e) {
          await client.createThread(assistantId, { metadata: { created_by: "ui" } });
        }
        
        // Fetch existing messages
        await fetchMessages();
        
        // Connect to EventSource for real-time updates
        const es = new EventSource(
          `${apiUrl || "/"}assistants/${assistantId}/threads/${activeThread.id}/messages/stream`
        );
        
        es.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === "message") {
              const newMessage: Message = {
                id: data.id,
                type: data.message_type === "human" 
                  ? "human" 
                  : data.message_type === "ai" 
                    ? "ai" 
                    : data.message_type === "tool" 
                      ? "tool" 
                      : data.message_type === "tool_result" 
                        ? "tool_result" 
                        : "system",
                content: data.content,
                metadata: data.metadata,
              };
              
              setMessages((prevMessages) => {
                // Check if message already exists
                const exists = prevMessages.some((m) => m.id === newMessage.id);
                if (exists) return prevMessages;
                return [...prevMessages, newMessage];
              });
            } else if (data.type === "run_update") {
              if (data.status === "in_progress") {
                setActiveThread((prev) => 
                  prev ? { ...prev, runId: data.id, runInProgress: true } : prev
                );
              } else if (data.status === "completed" || data.status === "failed") {
                setActiveThread((prev) => 
                  prev ? { ...prev, runInProgress: false } : prev
                );
              }
            }
          } catch (err) {
            console.error("Error parsing event data:", err, event.data);
          }
        };
        
        es.onerror = (e) => {
          console.error("EventSource error:", e);
          es.close();
        };
        
        setEventSource(es);
        
        return () => {
          es.close();
        };
      } catch (err) {
        console.error("Error initializing thread:", err);
        setError(err instanceof Error ? err : new Error("Failed to initialize thread"));
      }
    };
    
    initializeThread();
    
    return () => {
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [activeThread?.id, assistantId, apiUrl]);

  // Function to send a message
  const sendMessage = async (messageContent: string) => {
    if (!activeThread || !assistantId) {
      toast.error("No active thread or assistant ID");
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      // Send the human message
      await client.createMessage(assistantId, activeThread.id, {
        content: messageContent,
        type: "human",
      });
      
      // Start a run
      const run = await client.createRun(assistantId, activeThread.id);
      
      // Update thread with run info
      setActiveThread({
        ...activeThread,
        runId: run.id,
        runInProgress: true,
      });
    } catch (err) {
      console.error("Error sending message:", err);
      setError(err instanceof Error ? err : new Error("Failed to send message"));
      toast.error("Failed to send message");
    } finally {
      setIsLoading(false);
    }
  };

  // Function to interrupt the current run
  const interrupt = async () => {
    if (!activeThread?.runId || !assistantId) {
      toast.error("No active run to interrupt");
      return;
    }
    
    try {
      await client.interrupt(assistantId, activeThread.id, activeThread.runId);
      
      // Add an interrupt message
      const interruptMessage: Message = {
        id: `interrupt-${Date.now()}`,
        type: "interrupt",
        content: "Message interrupted by user",
      };
      
      setMessages((prev) => [...prev, interruptMessage]);
      
      setActiveThread({
        ...activeThread,
        runInProgress: false,
      });
      
      toast.success("Message interrupted");
    } catch (err) {
      console.error("Error interrupting run:", err);
      toast.error("Failed to interrupt message");
    }
  };

  const value = {
    messages,
    sendMessage,
    isLoading,
    error,
    interrupt,
    client,
  };

  return (
    <StreamContext.Provider value={value}>
      {children}
    </StreamContext.Provider>
  );
}

export function useStream() {
  const context = useContext(StreamContext);
  if (context === undefined) {
    throw new Error("useStream must be used within a StreamProvider");
  }
  return context;
}