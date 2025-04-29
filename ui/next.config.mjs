/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'standalone', // Important for containerized environments
  images: {
    domains: [],
  },
  // Configure server options
  serverRuntimeConfig: {
    port: parseInt(process.env.PORT, 10) || 3000,
  },
  experimental: {
    // Enable for optimized container size
    outputFileTracingRoot: process.env.NODE_ENV === 'production' ? undefined : process.cwd(),
  },
  // Allow env variables to be accessed in the browser when prefixed with NEXT_PUBLIC_
  env: {
    // You can add env vars here if needed
  },
}

export default nextConfig;
