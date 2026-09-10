import os
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace

# Sett opp Azure Monitor OpenTelemetry automatisk
# (Hentar koplingstrengen automatisk frå miljøvariabelen APPLICATIONINSIGHTS_CONNECTION_STRING)
configure_azure_monitor()

tracer = trace.get_tracer("min-azure-tjeneste")

# Sporet blir automatisk sendt til Azure Application Insights
with tracer.start_as_current_span("behandle_ordre") as span:
    span.set_attribute("ordre.id", "99887")
    print("Ordre behandlet og telemetridata sendt til Azure Monitor!")