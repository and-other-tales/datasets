import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

/**
 * API route handler for getting the current agent status
 */
export async function GET(request: NextRequest) {
  try {
    // Get the status endpoint URL from environment variable
    const statusUrl = process.env.DATASET_AGENT_STATUS_URL || '/status';
    
    // Convert relative URLs to absolute URLs for server-side API calls
    const apiUrl = statusUrl.startsWith('/') && !statusUrl.startsWith('//') ? 
      `http://localhost:${process.env.PORT || 8080}${statusUrl}` : statusUrl;
    
    try {
      // Query the Python agent status
      const response = await fetch(apiUrl, {
        method: 'GET',
        // Add reasonable timeout to avoid hanging requests
        signal: AbortSignal.timeout(5000),
      });

      if (!response.ok) {
        throw new Error(`Python agent returned status ${response.status}`);
      }

      const data = await response.json();
      
      // Ensure provider name is normalized for consistent UI handling
      if (data.provider) {
        data.provider = data.provider.toLowerCase();
        
        // Retrieve user preferences from localStorage if in browser
        if (typeof window !== 'undefined') {
          const savedProvider = localStorage.getItem('llm_provider_selection');
          const savedModel = localStorage.getItem('llm_model_selection');
          
          // If preferences exist but differ from server, prefer the saved preferences
          // This ensures UI stays consistent with user choices
          if (savedProvider && savedProvider !== data.provider) {
            console.log(`Provider mismatch: ${data.provider} (server) vs ${savedProvider} (saved)`);
            data.client_provider = savedProvider;
          }
          
          if (savedModel && savedModel !== data.model) {
            console.log(`Model mismatch: ${data.model} (server) vs ${savedModel} (saved)`);
            data.client_model = savedModel;
          }
        }
      }
      
      return NextResponse.json(data);
    } catch (error) {
      console.error('Failed to connect to Python agent status endpoint:', error);
      
      // Try to get saved preferences from localStorage
      let provider = 'bedrock';
      let model = 'anthropic.claude-3-7-sonnet-20250219-v1:0';
      
      if (typeof window !== 'undefined') {
        const savedProvider = localStorage.getItem('llm_provider_selection');
        const savedModel = localStorage.getItem('llm_model_selection');
        
        if (savedProvider) {
          provider = savedProvider;
        }
        
        if (savedModel) {
          model = savedModel;
        }
      }
      
      // Return a fallback response when the Python agent is unavailable
      return NextResponse.json({
        status: 'unavailable',
        provider: provider,
        model: model,
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