# AWS Reference Architecture

- AWS IoT Core for device ingestion.
- MSK or Kinesis for event streaming.
- EKS for control-plane services.
- RDS PostgreSQL for event store and audit metadata.
- Bedrock or OpenAI API for governed reasoning.
- ECR for edge and service images.
- Managed Grafana, CloudWatch, and OpenTelemetry Collector for observability.
- IAM Identity Center for RBAC.
- Secrets Manager and KMS for secrets and signing keys.
