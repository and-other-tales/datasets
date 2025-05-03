# Deploying to Google Cloud Run

This guide explains how to deploy the OtherTales Dataset Creator to Google Cloud Run.

## Prerequisites

1. A Google Cloud Platform (GCP) account
2. Google Cloud CLI (`gcloud`) installed and configured
3. Docker installed on your local machine
4. Docker Registry (like Google Container Registry or Docker Hub) access

## Building and Pushing the Docker Image

1. Build the Docker image locally:

```bash
docker build -t gcr.io/[PROJECT_ID]/dataset-creator:latest .
```

2. Push the image to Google Container Registry:

```bash
docker push gcr.io/[PROJECT_ID]/dataset-creator:latest
```

## Deploying to Cloud Run

1. Deploy the image to Cloud Run with startup probe configuration:

```bash
gcloud run deploy dataset-creator \
  --image gcr.io/[PROJECT_ID]/dataset-creator:latest \
  --platform managed \
  --region [REGION] \
  --allow-unauthenticated \
  --port 2024 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10 \
  --concurrency 80 \
  --startup-probe-path /startup \
  --startup-probe-period 5s \
  --startup-probe-timeout 3s \
  --startup-probe-failure-threshold 5
```

The startup probe configuration ensures that Cloud Run only routes traffic to your container once it's fully initialized and ready to receive requests. This is especially important for LangGraph applications which may need time to initialize models and connections.

2. Set environment variables needed for your deployment:

```bash
gcloud run services update dataset-creator \
  --set-env-vars="LLM_PROVIDER=bedrock,BEDROCK_MODEL_ID=anthropic.claude-3-7-sonnet-20250219-v1:0,PYTHONPATH=/app" \
  --region [REGION]
```

3. For secrets like API keys, use Secret Manager:

```bash
# Create a secret
gcloud secrets create aws-access-key --replication-policy="automatic" --data-file="./access-key.txt"

# Assign to Cloud Run service
gcloud run services update dataset-creator \
  --add-volume-secret=aws-key-mount:aws-access-key:latest \
  --update-env-vars="AWS_ACCESS_KEY_ID=/aws-key-mount" \
  --region [REGION]
```

## Post-Deployment Verification

1. Access your deployed service at the URL provided by Cloud Run
2. Verify that the API endpoints are working correctly
3. Check logs for any errors:

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=dataset-creator" --limit 10
```

## Troubleshooting Common Issues

1. **ImportError with relative imports**: Make sure your `PYTHONPATH` is set correctly in the Dockerfile and your imports use the full package path.

2. **Module not found errors**: Verify that your dependencies are properly installed in the Docker image.

3. **Timeout issues**: Adjust the Cloud Run timeout settings if your operations take longer than the default timeout.

4. **Memory errors**: Increase the memory allocation if you're processing large datasets or crawling many pages.

5. **Cold start problems**: Consider setting min-instances to 1 to avoid cold starts if responsiveness is critical.

6. **Container errors in logs**: Check that your entrypoint.sh script is properly configured and has execute permissions.

7. **Startup probe failures**: If your service fails to start due to startup probe failures:
   - Check the `/startup` endpoint is correctly implemented and responding with 200 status code
   - Increase the `startup-probe-failure-threshold` if your application needs more time to initialize
   - Inspect the logs using `gcloud logging read` to see why the startup probe is failing
   - Ensure all required environment variables are properly set
   - Test the startup endpoint locally by building and running the container:
     ```bash
     docker build -t dataset-creator:test .
     docker run -p 2024:2024 dataset-creator:test
     curl http://localhost:2024/startup
     ```