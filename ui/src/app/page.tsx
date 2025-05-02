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
    // Fetch current provider info
    const fetchProviderInfo = async () => {
      try {
        const response = await fetch('/api/agent/status');
        let providerInfo = "AWS Bedrock Claude";
        
        if (response.ok) {
          const data = await response.json();
          if (data.provider && data.model) {
            const providerName = data.provider.charAt(0).toUpperCase() + data.provider.slice(1);
            providerInfo = `${providerName} (${data.model})`;
          }
        }
        
        // Add initial welcome message with provider info
        setMessages([
          {
            id: uuidv4(),
            role: "ai",
            content: `# Welcome to ${agentName}

I'm an AI assistant specialized in helping you create high-quality datasets from web content.
Currently using: ${providerInfo}

I can help you:
- Crawl websites with configurable depth and filters
- Convert HTML to markdown text
- Create structured datasets
- Push datasets to the HuggingFace Hub

To get started, tell me which website you'd like to crawl or what kind of dataset you want to create.
You can also change the LLM provider using the dropdown in the top right.`,
            timestamp: new Date(),
          },
        ]);
      } catch (error) {
        console.error("Error fetching provider info:", error);
        // Fallback to default message if provider info can't be fetched
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
      }
    };
    
    fetchProviderInfo();

    // Set up message listener
    const unsubscribe = agentClient.addMessageListener((message) => {
      setMessages((prevMessages) => [...prevMessages, message]);
      setIsLoading(false);
    });
    
    // Set up polling to detect environment variable changes
    const statusPollInterval = setInterval(async () => {
      try {
        const response = await fetch('/api/agent/status');
        if (response.ok) {
          const data = await response.json();
          // Update stored preferences if environment variables have changed
          if (typeof window !== 'undefined' && data.provider && data.model) {
            const savedProvider = localStorage.getItem('llm_provider_selection');
            const savedModel = localStorage.getItem('llm_model_selection');
            
            // If env var changed externally, update local storage
            if (savedProvider !== data.provider || savedModel !== data.model) {
              localStorage.setItem('llm_provider_selection', data.provider);
              localStorage.setItem('llm_model_selection', data.model);
              
              // Optionally notify the user
              if (messages.length > 0) {
                const providerName = data.provider.charAt(0).toUpperCase() + data.provider.slice(1);
                setMessages((prevMessages) => [
                  ...prevMessages, 
                  {
                    id: uuidv4(),
                    role: "system",
                    content: `The LLM provider has been updated to ${providerName} (${data.model}).`,
                    timestamp: new Date(),
                  }
                ]);
              }
            }
          }
        }
      } catch (error) {
        console.error("Error polling status:", error);
      }
    }, 30000); // Poll every 30 seconds
    
    return () => {
      unsubscribe();
      agentClient.disconnect();
      clearInterval(statusPollInterval);
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