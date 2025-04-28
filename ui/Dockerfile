FROM node:22-alpine AS builder

WORKDIR /app

# Install dependencies based on package-lock.json
COPY package*.json ./
RUN npm install --legacy-peer-deps

# Copy all files
COPY . .

# Make sure src/lib directory exists
RUN mkdir -p /app/src/lib/

# Run prebuild script to ensure all files exist
RUN chmod +x prebuild.sh
RUN sh -c "./prebuild.sh"

# Debug directories to ensure files exist
RUN ls -la /app/src/lib/

# Clean any previous builds and build with increased memory
RUN npm run clean || true
RUN NODE_OPTIONS=--max-old-space-size=4096 npm run build

# Runner stage
FROM node:22-alpine AS runner

WORKDIR /app

# Set environment variables
ENV NODE_ENV production
ENV PORT 8080
ENV NEXT_TELEMETRY_DISABLED 1

# Set up non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs && \
    chown -R nextjs:nodejs /app

# Copy built app
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Create directory for GCS mount and set permissions
RUN mkdir -p /gcs && chown nextjs:nodejs /gcs

USER nextjs

# Expose port
EXPOSE 8080

# Start server
CMD ["node", "server.js"]