import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  // Return UI configuration
  return NextResponse.json({
    appName: process.env.NEXT_PUBLIC_AGENT_NAME || "OtherTales Datasets Agent",
    provider: process.env.LLM_PROVIDER || "Unknown",
    version: "1.0.0",
  });
}