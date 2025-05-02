"use client";

import React, { createContext, useContext, useState } from "react";

export type Artifact = {
  id: string;
  name: string;
  content: string;
  type: string;
  created_at: string;
};

interface ArtifactContextType {
  artifacts: Artifact[];
  addArtifact: (artifact: Artifact) => void;
  clearArtifacts: () => void;
}

const ArtifactContext = createContext<ArtifactContextType | undefined>(undefined);

export function ArtifactProvider({ children }: { children: React.ReactNode }) {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);

  const addArtifact = (artifact: Artifact) => {
    setArtifacts((prev) => [...prev, artifact]);
  };

  const clearArtifacts = () => {
    setArtifacts([]);
  };

  return (
    <ArtifactContext.Provider value={{ artifacts, addArtifact, clearArtifacts }}>
      {children}
    </ArtifactContext.Provider>
  );
}

export function useArtifacts() {
  const context = useContext(ArtifactContext);
  if (context === undefined) {
    throw new Error("useArtifacts must be used within an ArtifactProvider");
  }
  return context;
}