import { safeJsonParse } from './utils';

interface AgentConfig {
  crawlDepth?: number;
  maxPages?: number;
  allowedDomains?: string[];
  excludeUrls?: string[];
  extractionRules?: Record<string, string>;
}

interface AgentResponse {
  status: 'success' | 'error';
  message: string;
  data?: any;
}

/**
 * Runs the dataset agent with a message
 */
export async function runAgent(message: string): Promise<AgentResponse> {
  try {
    const endpoint = process.env.NEXT_PUBLIC_AGENT_API_URL || '/api/agent';
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return {
      status: 'success',
      message: data.message || 'Agent executed successfully',
      data,
    };
  } catch (error) {
    console.error('Error running agent:', error);
    return {
      status: 'error',
      message: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
}

/**
 * Connects to the LangGraph server
 */
export async function connectToServer(serverUrl?: string): Promise<AgentResponse> {
  try {
    const url = serverUrl || process.env.NEXT_PUBLIC_LANGGRAPH_URL || '/api/connect';
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return {
      status: 'success',
      message: 'Connected to LangGraph server',
      data,
    };
  } catch (error) {
    console.error('Error connecting to server:', error);
    return {
      status: 'error',
      message: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
}

/**
 * Updates the agent configuration
 */
export async function updateAgentConfig(config: AgentConfig): Promise<AgentResponse> {
  try {
    const endpoint = process.env.NEXT_PUBLIC_AGENT_CONFIG_URL || '/api/config';
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return {
      status: 'success',
      message: 'Agent configuration updated',
      data,
    };
  } catch (error) {
    console.error('Error updating agent config:', error);
    return {
      status: 'error',
      message: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
}