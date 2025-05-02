import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const langgraphUrl = process.env.DATASET_AGENT_URL || "http://localhost:2024/agent";
  
  try {
    // Fetch available assistants
    const assistantsResponse = await fetch(`${langgraphUrl}/assistants`);
    
    if (!assistantsResponse.ok) {
      throw new Error(`Failed to fetch assistants: ${assistantsResponse.statusText}`);
    }
    
    const assistantsData = await assistantsResponse.json();
    
    // We should have at least one assistant
    if (!assistantsData.data || assistantsData.data.length === 0) {
      return NextResponse.json(
        { error: "No assistants available" },
        { status: 404 }
      );
    }
    
    // Return the first assistant (usually there's only one in LangGraph)
    const assistant = assistantsData.data[0];
    
    return NextResponse.json({
      assistantId: assistant.id,
      assistantName: assistant.name || "Dataset Agent",
      apiUrl: langgraphUrl,
    });
  } catch (error) {
    console.error("Error connecting to LangGraph:", error);
    return NextResponse.json(
      { error: "Failed to connect to LangGraph server" },
      { status: 500 }
    );
  }
}