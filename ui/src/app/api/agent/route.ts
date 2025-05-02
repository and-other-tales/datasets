import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

/**
 * API route handler for dataset agent interactions
 */
export async function POST(request: NextRequest) {
  try {
    // Parse the request body
    const body = await request.json();
    const { message, thread_id } = body;
    
    if (!message) {
      return NextResponse.json({ error: 'Message is required' }, { status: 400 });
    }

    // Try to connect to the Python agent
    // When deployed with Cloud Run and nginx reverse proxy, the agent is accessible at /agent
    // For local development, connect to LangGraph server directly
    let agentUrl = process.env.DATASET_AGENT_URL || 'http://localhost:2024/agent';
    
    // If agentUrl doesn't start with http:// or https://, assume it's a relative path and prepend http://localhost:2024
    if (!agentUrl.startsWith('http://') && !agentUrl.startsWith('https://')) {
      agentUrl = `http://localhost:2024${agentUrl.startsWith('/') ? agentUrl : '/' + agentUrl}`;
    }
    
    console.log(`Connecting to dataset agent at ${agentUrl}`);
    
    try {
      // Forward the request to the Python agent with thread_id if available
      const requestBody: Record<string, string> = { message };
      if (thread_id) {
        requestBody.thread_id = thread_id;
      }
      
      const response = await fetch(agentUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        // Add reasonable timeout to avoid hanging requests
        signal: AbortSignal.timeout(15000)
      });

      if (!response.ok) {
        throw new Error(`Python agent returned status ${response.status}`);
      }

      const data = await response.json();
      return NextResponse.json(data);
    } catch (error) {
      console.error('Failed to connect to Python agent:', error);
      
      // Return a helpful response with instructions
      return NextResponse.json({
        message: `## Agent Connection Issue

I'm unable to reach the agent backend that handles dataset creation. Here's what you can do:

1. Make sure the agent is running by executing:
   \`\`\`bash
   python agent_standalone.py
   \`\`\`

2. Or use the start script that launches both the UI and agent:
   \`\`\`bash
   ./start.sh
   \`\`\`

3. If you continue to have issues, check that port 2024 is available and not blocked by firewall settings.

You can reload this page once the agent is running.`,
        status: 'fallback',
        thread_id
      });
    }
  } catch (error) {
    console.error('Error in agent API route:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}