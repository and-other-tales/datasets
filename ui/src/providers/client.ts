import type { CreateMessage } from "@langchain/langgraph-sdk";

export interface InboxAction {
  id: string;
  input: unknown;
  status: "PENDING" | "STARTING" | "RUNNING" | "COMPLETE" | "ERROR";
  error?: Error;
  result?: unknown;
}

export interface Client {
  listAssistants: () => Promise<
    Array<{
      id: string;
      name: string;
      description: string;
      metadata: Record<string, unknown>;
    }>
  >;
  listThreads: (assistantId: string) => Promise<Array<{ id: string }>>;
  getMessages: (
    assistantId: string,
    threadId: string,
  ) => Promise<Array<{ id: string; type: string; content: unknown; metadata?: Record<string, unknown> }>>;
  getRun: (
    assistantId: string,
    threadId: string,
    runId: string,
  ) => Promise<{ status: string; error?: string }>;
  getThread: (
    assistantId: string,
    threadId: string,
  ) => Promise<{ metadata?: Record<string, unknown> }>;
  createThread: (
    assistantId: string,
    data?: { metadata?: Record<string, unknown> },
  ) => Promise<{ id: string }>;
  createMessage: (
    assistantId: string,
    threadId: string,
    data: CreateMessage,
  ) => Promise<{ id: string }>;
  createRun: (
    assistantId: string,
    threadId: string,
    data?: { metadata?: Record<string, unknown> },
  ) => Promise<{ id: string }>;
  interrupt: (
    assistantId: string,
    threadId: string,
    runId: string,
  ) => Promise<void>;
}

export class LangGraphClient implements Client {
  baseUrl: string;

  constructor(baseUrl: string = "/") {
    this.baseUrl = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  }

  async listAssistants() {
    const response = await fetch(`${this.baseUrl}assistants`);
    if (!response.ok) {
      throw new Error(`Failed to list assistants: ${response.statusText}`);
    }
    const data = await response.json();
    return data.data;
  }

  async listThreads(assistantId: string) {
    const response = await fetch(
      `${this.baseUrl}assistants/${assistantId}/threads`,
    );
    if (!response.ok) {
      throw new Error(`Failed to list threads: ${response.statusText}`);
    }
    const data = await response.json();
    return data.data;
  }

  async getMessages(assistantId: string, threadId: string) {
    const response = await fetch(
      `${this.baseUrl}assistants/${assistantId}/threads/${threadId}/messages`,
    );
    if (!response.ok) {
      throw new Error(`Failed to get messages: ${response.statusText}`);
    }
    const data = await response.json();
    return data.data;
  }

  async getRun(assistantId: string, threadId: string, runId: string) {
    const response = await fetch(
      `${this.baseUrl}assistants/${assistantId}/threads/${threadId}/runs/${runId}`,
    );
    if (!response.ok) {
      throw new Error(`Failed to get run: ${response.statusText}`);
    }
    const data = await response.json();
    return data;
  }

  async getThread(assistantId: string, threadId: string) {
    const response = await fetch(
      `${this.baseUrl}assistants/${assistantId}/threads/${threadId}`,
    );
    if (!response.ok) {
      throw new Error(`Failed to get thread: ${response.statusText}`);
    }
    const data = await response.json();
    return data;
  }

  async createThread(
    assistantId: string,
    data?: { metadata?: Record<string, unknown> },
  ) {
    const response = await fetch(
      `${this.baseUrl}assistants/${assistantId}/threads`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data || {}),
      },
    );
    if (!response.ok) {
      throw new Error(`Failed to create thread: ${response.statusText}`);
    }
    const responseData = await response.json();
    return responseData;
  }

  async createMessage(
    assistantId: string,
    threadId: string,
    data: CreateMessage,
  ) {
    const response = await fetch(
      `${this.baseUrl}assistants/${assistantId}/threads/${threadId}/messages`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      },
    );
    if (!response.ok) {
      throw new Error(`Failed to create message: ${response.statusText}`);
    }
    const responseData = await response.json();
    return responseData;
  }

  async createRun(
    assistantId: string,
    threadId: string,
    data?: { metadata?: Record<string, unknown> },
  ) {
    const response = await fetch(
      `${this.baseUrl}assistants/${assistantId}/threads/${threadId}/runs`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data || {}),
      },
    );
    if (!response.ok) {
      throw new Error(`Failed to create run: ${response.statusText}`);
    }
    const responseData = await response.json();
    return responseData;
  }

  async interrupt(assistantId: string, threadId: string, runId: string) {
    const response = await fetch(
      `${this.baseUrl}assistants/${assistantId}/threads/${threadId}/runs/${runId}/interrupt`,
      {
        method: "POST",
      },
    );
    if (!response.ok) {
      throw new Error(`Failed to interrupt run: ${response.statusText}`);
    }
  }
}