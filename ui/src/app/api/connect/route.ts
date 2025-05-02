import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

/**
 * API route handler for checking connection to the Python agent
 */
export async function POST(request: NextRequest) {
  try {
    // Try to connect to the Python agent via nginx reverse proxy
    // Get the agent URL from environment variable
    let agentUrl = process.env.DATASET_AGENT_URL || '/agent';
    // Replace /agent with /status for status checks if needed
    let statusUrl = process.env.DATASET_AGENT_STATUS_URL || agentUrl.replace('/agent', '/status');
    
    // Convert relative URLs to absolute URLs for server-side API calls
    const apiUrl = statusUrl.startsWith('/') && !statusUrl.startsWith('//') ? 
      `http://localhost:${process.env.PORT || 8080}${statusUrl}` : statusUrl;
    
    try {
      // Check if the Python agent is running
      console.log(`Checking agent status at ${apiUrl}`);
      const response = await fetch(apiUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        // Add reasonable timeout to avoid hanging requests
        signal: AbortSignal.timeout(5000),
      });

      if (!response.ok) {
        throw new Error(`Python agent returned status ${response.status}`);
      }

      const data = await response.json();
      return NextResponse.json({
        connected: true,
        status: data.status || 'running',
        message: 'Connected to dataset agent',
        persistence: data.persistence || false
      });
    } catch (error) {
      console.error('Failed to connect to Python agent:', error);
      
      // Return a connection failure response
      return NextResponse.json({
        connected: false,
        status: 'unavailable',
        message: 'Dataset agent is not running. Please start with python dataset_agent.py --api'
      });
    }
  } catch (error) {
    console.error('Error in connect API route:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}