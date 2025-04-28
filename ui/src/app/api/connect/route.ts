import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

/**
 * API route handler for checking connection to the Python agent
 */
export async function POST(request: NextRequest) {
  try {
    // Try to connect to the Python agent
    const agentUrl = process.env.DATASET_AGENT_URL || 'http://localhost:8080/status';
    
    try {
      // Check if the Python agent is running
      const response = await fetch(agentUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
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