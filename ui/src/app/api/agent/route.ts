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
    const agentUrl = process.env.DATASET_AGENT_URL || 'http://localhost:8080/agent';
    
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
      });

      if (!response.ok) {
        throw new Error(`Python agent returned status ${response.status}`);
      }

      const data = await response.json();
      return NextResponse.json(data);
    } catch (error) {
      console.error('Failed to connect to Python agent:', error);
      
      // Return a fallback response when the Python agent is unavailable
      return NextResponse.json({
        message: "I'm a placeholder response. The real agent is not connected. Please start the Python agent with 'python dataset_agent.py --api'.",
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