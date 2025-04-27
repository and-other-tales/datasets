import { v4 as uuidv4 } from 'uuid';

export interface Message {
  id: string;
  role: "human" | "ai" | "system" | "tool";
  content: string;
  timestamp: Date;
}

export class AgentClient {
  private messages: Message[] = [];
  private listeners: ((message: Message) => void)[] = [];
  private static instance: AgentClient;
  private threadId: string | null = null;

  constructor() {
    if (AgentClient.instance) {
      return AgentClient.instance;
    }
    
    // Only access localStorage in browser environment
    if (typeof window !== 'undefined') {
      // Generate a unique thread ID for this session
      this.threadId = localStorage.getItem('agent_thread_id') || uuidv4();
      localStorage.setItem('agent_thread_id', this.threadId);
    } else {
      // For SSR, just create a new ID but don't save it
      this.threadId = uuidv4();
    }
    
    AgentClient.instance = this;
  }

  static getInstance(): AgentClient {
    if (!AgentClient.instance) {
      AgentClient.instance = new AgentClient();
    }
    return AgentClient.instance;
  }

  getThreadId(): string | null {
    return this.threadId;
  }

  setThreadId(threadId: string): void {
    this.threadId = threadId;
    if (typeof window !== 'undefined') {
      localStorage.setItem('agent_thread_id', threadId);
    }
  }

  async sendMessage(content: string): Promise<void> {
    try {
      // Add user message
      const userMessage: Message = {
        id: uuidv4(),
        role: 'human',
        content,
        timestamp: new Date(),
      };
      
      this.messages.push(userMessage);
      
      // Import the runAgent function
      const { runAgent } = await import('./agent-integration');
      
      try {
        // Send to real agent if available with thread ID for state persistence
        const response = await runAgent(content, this.threadId || undefined);
        
        // If the response includes a thread_id, store it
        if (response.thread_id) {
          this.setThreadId(response.thread_id);
        }
        
        const assistantMessage: Message = {
          id: uuidv4(),
          role: 'ai',
          content: response.status === 'success' 
            ? response.message 
            : 'Sorry, there was an error connecting to the agent. Using placeholder response.',
          timestamp: new Date(),
        };
        
        this.messages.push(assistantMessage);
        this.notifyListeners(assistantMessage);
      } catch (agentError) {
        console.error("Failed to connect to agent, using fallback:", agentError);
        
        // Fallback to simulation if agent connection fails
        const assistantMessage: Message = {
          id: uuidv4(),
          role: 'ai',
          content: 'This is a placeholder response. The dataset agent is currently not available.',
          timestamp: new Date(),
        };
        
        this.messages.push(assistantMessage);
        this.notifyListeners(assistantMessage);
      }
    } catch (error) {
      console.error("Error sending message:", error);
      throw error;
    }
  }

  addMessageListener(callback: (message: Message) => void): () => void {
    this.listeners.push(callback);
    return () => {
      this.listeners = this.listeners.filter(listener => listener !== callback);
    };
  }

  private notifyListeners(message: Message): void {
    this.listeners.forEach(listener => listener(message));
  }

  getMessages(): Message[] {
    return this.messages;
  }

  clearMessages(): void {
    this.messages = [];
  }
  
  resetThread(): void {
    this.threadId = uuidv4();
    if (typeof window !== 'undefined') {
      localStorage.setItem('agent_thread_id', this.threadId);
    }
    this.clearMessages();
  }
  
  disconnect(): void {
    this.listeners = [];
  }
}