"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";

export type ThreadInfo = {
  id: string;
  runId?: string;
  runInProgress: boolean;
};

type ThreadContextType = {
  activeThread: ThreadInfo | null;
  setActiveThread: React.Dispatch<
    React.SetStateAction<ThreadInfo | null>
  >;
  createNewThread: () => ThreadInfo;
};

const ThreadContext = createContext<ThreadContextType | undefined>(undefined);

export function ThreadProvider({ children }: { children: React.ReactNode }) {
  const [activeThread, setActiveThread] = useState<ThreadInfo | null>(null);
  
  // Create a new thread with a unique ID
  const createNewThread = () => {
    const newThread = {
      id: uuidv4(),
      runInProgress: false,
    };
    setActiveThread(newThread);
    return newThread;
  };

  // Set up an initial thread on first load
  useEffect(() => {
    if (!activeThread) {
      createNewThread();
    }
  }, []);

  return (
    <ThreadContext.Provider
      value={{ activeThread, setActiveThread, createNewThread }}
    >
      {children}
    </ThreadContext.Provider>
  );
}

export function useThread() {
  const context = useContext(ThreadContext);
  if (context === undefined) {
    throw new Error("useThread must be used within a ThreadProvider");
  }
  return context;
}