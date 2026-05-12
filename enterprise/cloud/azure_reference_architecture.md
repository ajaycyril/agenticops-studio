# Azure Reference Architecture

- IoT Hub or Event Hubs for device/event ingestion.
- Azure Kubernetes Service for control-plane services.
- Azure Database for PostgreSQL for event store and audit metadata.
- Azure OpenAI or OpenAI API for governed reasoning.
- Azure Container Registry for edge model packages.
- Managed Grafana, Monitor, and Log Analytics for observability.
- Entra ID for enterprise identity and RBAC.
- Key Vault for secrets and signing keys.
