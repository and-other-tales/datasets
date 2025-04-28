import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

/**
 * API route handler for dataset agent configuration
 */
export async function POST(request: NextRequest) {
  try {
    // Parse the request body
    const config = await request.json();
    
    // Try to connect to the Python agent
    const configUrl = process.env.DATASET_AGENT_CONFIG_URL || 'http://localhost:8080/config';
    
    try {
      // Forward the config to the Python agent
      console.log(`Updating agent config at ${configUrl}`, config);
      const response = await fetch(configUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
        // Add reasonable timeout to avoid hanging requests
        signal: AbortSignal.timeout(10000),
      });

      if (!response.ok) {
        throw new Error(`Python agent returned status ${response.status}`);
      }

      const data = await response.json();
      return NextResponse.json(data);
    } catch (error) {
      console.error('Failed to connect to Python agent config:', error);
      
      // Return a fallback response when the Python agent is unavailable
      return NextResponse.json({
        message: "Configuration update failed: Python agent is not connected",
        status: 'error'
      });
    }
  } catch (error) {
    console.error('Error in config API route:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}