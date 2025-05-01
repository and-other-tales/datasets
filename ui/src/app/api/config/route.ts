import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

/**
 * API route handler for dataset agent configuration
 */
export async function POST(request: NextRequest) {
  try {
    // Parse the request body
    const body = await request.json();
    
    // Process LLM configuration
    const llmConfig: Record<string, any> = {};
    
    // Handle LLM provider change
    if (body.provider) {
      llmConfig.llm_provider = body.provider.toLowerCase();
      
      // Handle provider-specific settings
      if (body.model) {
        llmConfig.model = body.model;
      }
      
      if (body.temperature !== undefined) {
        llmConfig.temperature = parseFloat(body.temperature);
      }
      
      if (body.maxTokens !== undefined) {
        llmConfig.max_tokens = parseInt(body.maxTokens, 10);
      }
      
      // For Azure, allow setting deployment name
      if (llmConfig.llm_provider === 'azure' && body.deployment) {
        llmConfig.deployment = body.deployment;
      }
    }
    
    // Handle any other general configurations
    const configToSend = {
      ...body,
      ...llmConfig
    };
    
    // Try to connect to the Python agent
    const configUrl = process.env.DATASET_AGENT_CONFIG_URL || 'http://localhost:8080/config';
    
    try {
      // Forward the config to the Python agent
      console.log(`Updating agent config at ${configUrl}`, configToSend);
      const response = await fetch(configUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(configToSend),
        // Add reasonable timeout to avoid hanging requests
        signal: AbortSignal.timeout(10000),
      });

      if (!response.ok) {
        throw new Error(`Python agent returned status ${response.status}`);
      }

      const data = await response.json();
      return NextResponse.json({
        ...data,
        processed_config: configToSend
      });
    } catch (error) {
      console.error('Failed to connect to Python agent config:', error);
      
      // Return a fallback response when the Python agent is unavailable
      return NextResponse.json({
        message: "Configuration update failed: Python agent is not connected",
        status: 'error',
        attempted_config: configToSend
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