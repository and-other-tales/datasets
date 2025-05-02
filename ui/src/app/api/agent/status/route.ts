import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const langgraphUrl = process.env.DATASET_AGENT_URL || "http://localhost:2024/agent";
  
  try {
    // Check LangGraph status
    const statusResponse = await fetch(`${langgraphUrl}/status`);
    
    if (!statusResponse.ok) {
      throw new Error(`Failed to fetch status: ${statusResponse.statusText}`);
    }
    
    const statusData = await statusResponse.json();
    
    return NextResponse.json({
      status: "online",
      langgraphStatus: statusData,
    });
  } catch (error) {
    console.error("Error checking LangGraph status:", error);
    return NextResponse.json(
      { 
        status: "offline",
        error: "LangGraph server is not responding" 
      },
      { status: 200 } // Still return 200 to let the UI handle the offline state
    );
  }
}