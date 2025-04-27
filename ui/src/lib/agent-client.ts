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

  constructor() {
    if (AgentClient.instance) {
      return AgentClient.instance;
    }
    AgentClient.instance = this;
  }

  static getInstance(): AgentClient {
    if (!AgentClient.instance) {
      AgentClient.instance = new AgentClient();
    }
    return AgentClient.instance;
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
      
      // Simulate agent response
      setTimeout(() => {
        const assistantMessage: Message = {
          id: uuidv4(),
          role: 'ai',
          content: 'This is a placeholder response from the agent.',
          timestamp: new Date(),
        };
        
        this.messages.push(assistantMessage);
        
        // Notify listeners
        this.notifyListeners(assistantMessage);
      }, 1000);
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
  
  disconnect(): void {
    this.listeners = [];
  }
}