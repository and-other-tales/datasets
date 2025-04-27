#!/bin/sh
set -e

# Create lib directory if it doesn't exist
echo "Checking src/lib directory..."
mkdir -p src/lib

# Make sure agent-client.ts exists
if [ ! -f "src/lib/agent-client.ts" ]; then
  echo "src/lib/agent-client.ts not found, creating..."
  cat > src/lib/agent-client.ts << 'EOF'
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
EOF
fi

# Make sure utils.ts exists
if [ ! -f "src/lib/utils.ts" ]; then
  echo "src/lib/utils.ts not found, creating..."
  cat > src/lib/utils.ts << 'EOF'
import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function getInitials(name: string): string {
  return name
    .split(' ')
    .map(part => part.charAt(0))
    .join('')
    .toUpperCase()
    .substring(0, 2);
}

export function formatMarkdown(text: string): string {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br />');
}

/**
 * Safely parses JSON, returning a default value if parsing fails
 */
export function safeJsonParse<T>(jsonString: string, defaultValue: T): T {
  try {
    return JSON.parse(jsonString) as T;
  } catch (error) {
    console.error('Error parsing JSON:', error);
    return defaultValue;
  }
}
EOF
fi

# Create index.ts for easier imports
if [ ! -f "src/lib/index.ts" ]; then
  echo "Creating src/lib/index.ts for re-exports..."
  cat > src/lib/index.ts << 'EOF'
// Re-export all module contents for easier imports
export * from './agent-client';
export * from './utils';
export * from './agent-integration';
EOF
fi

# List files in src/lib
echo "Files in src/lib:"
ls -la src/lib/

# Check if module resolution config files exist
echo "Checking for configuration files..."
for file in tsconfig.json jsconfig.json next.config.js next.config.mjs; do
  if [ -f "$file" ]; then
    echo "✅ Found $file"
  else
    echo "❌ Missing $file"
  fi
done

# Clean previous build artifacts
echo "Cleaning previous build..."
rm -rf .next out node_modules/.cache

echo "Setting up environment for build..."
export NODE_OPTIONS="--max-old-space-size=4096"

echo "Prebuild checks complete"
exit 0