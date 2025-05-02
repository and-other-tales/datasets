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

1. Deploy the image to Cloud Run:

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
  --concurrency 80
```

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