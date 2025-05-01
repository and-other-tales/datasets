import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

/**
 * API route handler for getting the current agent status
 */
export async function GET(request: NextRequest) {
  try {
    // Get the status endpoint URL from environment variable
    const statusUrl = process.env.DATASET_AGENT_STATUS_URL || 'http://localhost:8080/status';
    
    try {
      // Query the Python agent status
      const response = await fetch(statusUrl, {
        method: 'GET',
        // Add reasonable timeout to avoid hanging requests
        signal: AbortSignal.timeout(5000),
      });

      if (!response.ok) {
        throw new Error(`Python agent returned status ${response.status}`);
      }

      const data = await response.json();
      return NextResponse.json(data);
    } catch (error) {
      console.error('Failed to connect to Python agent status endpoint:', error);
      
      // Return a fallback response when the Python agent is unavailable
      return NextResponse.json({
        status: 'unavailable',
        provider: 'bedrock',
        model: 'anthropic.claude-3-7-sonnet-20250219-v1:0',
        message: "Unable to determine current LLM provider. Python agent is not connected."
      });
    }
  } catch (error) {
    console.error('Error in agent status API route:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}