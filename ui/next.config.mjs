/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  webpack: (config) => {
    // Handle web worker files
    config.resolve.fallback = { fs: false, net: false, tls: false };
    return config;
  },
  // Handle LangGraph API forwarding
  async rewrites() {
    return [
      {
        source: '/api/connect',
        destination: '/api/agent/connect',
      },
      {
        source: '/api/agent/status',
        destination: '/api/agent/status',
      },
    ];
  },
};

export default nextConfig;