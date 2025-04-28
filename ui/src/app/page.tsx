"use client";

import React, { useState, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import ChatInterface from "@/components/chat-interface";
import { AgentClient, Message } from "@/lib";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const agentClient = AgentClient.getInstance();
  
  // Get agent name from environment variable or use default
  const agentName = process.env.NEXT_PUBLIC_AGENT_NAME || "OtherTales Datasets Agent";
  
  useEffect(() => {
    // Try to connect to agent when the app loads
    const connectToAgent = async () => {
      try {
        const { connectToServer } = await import("@/lib/agent-integration");
        await connectToServer();
      } catch (error) {
        console.log("Could not connect to agent server:", error);
      }
    };
    
    connectToAgent();
  }, []);

  useEffect(() => {
    // Add initial welcome message
    setMessages([
      {
        id: uuidv4(),
        role: "ai",
        content: `# Welcome to ${agentName}

I'm an AI assistant specialized in helping you create high-quality datasets from web content.

I can help you:
- Crawl websites with configurable depth and filters
- Convert HTML to markdown text
- Create structured datasets
- Push datasets to the HuggingFace Hub

To get started, tell me which website you'd like to crawl or what kind of dataset you want to create.`,
        timestamp: new Date(),
      },
    ]);

    // Set up message listener
    const unsubscribe = agentClient.addMessageListener((message) => {
      setMessages((prevMessages) => [...prevMessages, message]);
      setIsLoading(false);
    });

    return () => {
      unsubscribe();
      agentClient.disconnect();
    };
  }, []);

  const handleSendMessage = async (content: string) => {
    try {
      setIsLoading(true);

      // Add user message to the chat
      const userMessage: Message = {
        id: uuidv4(),
        role: "human",
        content,
        timestamp: new Date(),
      };

      setMessages((prevMessages) => [...prevMessages, userMessage]);

      // Send message to agent
      await agentClient.sendMessage(content);

    } catch (error) {
      console.error("Error sending message:", error);
      setIsLoading(false);

      // Add error message
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          id: uuidv4(),
          role: "system",
          content: "Sorry, there was an error processing your request. Please try again.",
          timestamp: new Date(),
        },
      ]);
    }
  };

  return (
    <main className="flex flex-col h-full bg-background">
      <header className="border-b border-border p-4 shadow-sm">
        <h1 className="font-bold text-xl">{agentName}</h1>
        <p className="text-sm text-muted-foreground">
          Create datasets from web content
        </p>
      </header>
      
      <div className="flex-1 overflow-hidden">
        <ChatInterface
          onSendMessage={handleSendMessage}
          messages={messages}
          isLoading={isLoading}
          agentName={agentName}
        />
      </div>
    </main>
  );
}