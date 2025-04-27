/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone', // Important for containerized environments
  images: {
    domains: [],
  },
  // Configure server options
  serverRuntimeConfig: {
    port: parseInt(process.env.PORT, 10) || 8080,
  },
  // Allow env variables to be accessed in the browser when prefixed with NEXT_PUBLIC_
  env: {
    // You can add env vars here if needed
  },
}

module.exports = nextConfig