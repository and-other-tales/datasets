"use client";

import { Thread } from "@/components/thread";
import { StreamProvider } from "@/providers/Stream";
import { ThreadProvider } from "@/providers/Thread";
import { ArtifactProvider } from "@/components/thread/artifact";
import { Toaster } from "sonner";
import React, { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";

export default function HomePage(): React.ReactNode {
  const searchParams = useSearchParams();
  const [assistantId, setAssistantId] = useState<string | null>(null);
  const [apiUrl, setApiUrl] = useState<string | null>(null);
  const [chatHistoryOpen, setChatHistoryOpen] = useState<boolean>(false);

  useEffect(() => {
    const assistantIdParam = searchParams.get("assistantId");
    const apiUrlParam = searchParams.get("apiUrl") || "/";
    const chatHistoryOpenParam = searchParams.get("chatHistoryOpen");

    if (assistantIdParam) {
      setAssistantId(assistantIdParam);
      console.log("Assistant ID from URL:", assistantIdParam);
    }
    
    setApiUrl(apiUrlParam);
    setChatHistoryOpen(chatHistoryOpenParam === "true");
  }, [searchParams]);

  return (
    <React.Suspense fallback={<div className="p-8 text-center">Loading datasets agent...</div>}>
      <Toaster />
      <ThreadProvider>
        <StreamProvider apiUrl={apiUrl} assistantId={assistantId}>
          <ArtifactProvider>
            <div className="flex flex-col h-screen w-full">
              <header className="border-b border-border p-4 flex items-center justify-between">
                <h1 className="text-xl font-semibold">OtherTales Datasets Agent</h1>
                {assistantId && (
                  <div className="text-xs text-muted-foreground">
                    Connected to assistant: {assistantId.slice(0, 8)}...
                  </div>
                )}
              </header>
              <Thread showHistory={chatHistoryOpen} />
            </div>
          </ArtifactProvider>
        </StreamProvider>
      </ThreadProvider>
    </React.Suspense>
  );
}