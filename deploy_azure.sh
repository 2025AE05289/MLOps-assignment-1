#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy_azure.sh — Deploy Heart Disease Prediction API to Azure AKS
#
# Optimized for Azure Student Subscription ($100 credits, 4 vCPU quota):
# - Uses 1 node of Standard_B2s (2 vCPUs, 4GB RAM) -> ~$0.04/hour
# - Exposes API via Azure LoadBalancer with a Public IP
# - Includes clean teardown to preserve your credits
#
# Usage:
#   bash deploy_azure.sh deploy    # Create cluster and deploy API
#   bash deploy_azure.sh status    # Check cluster and pod status
#   bash deploy_azure.sh test      # Send test prediction to Azure Public IP
#   bash deploy_azure.sh destroy   # Delete resource group (STOPS ALL BILLING)
# ─────────────────────────────────────────────────────────────────────────────

set -e

RESOURCE_GROUP="${AZURE_RG:-rg-mlops-heart}"
LOCATION="${AZURE_LOCATION:-eastus}"
CLUSTER_NAME="${AZURE_CLUSTER:-aks-heart-mlops}"
VM_SIZE="Standard_B2s"   # 2 vCPUs - strictly safe for student account quota
NODE_COUNT=1

ACTION="${1:-help}"

case "$ACTION" in
  deploy)
    echo "============================================================"
    echo "  Deploying Heart Disease API to Azure Kubernetes Service"
    echo "============================================================"
    echo "Resource Group: $RESOURCE_GROUP"
    echo "Region:         $LOCATION"
    echo "Cluster Name:   $CLUSTER_NAME"
    echo "VM Size:        $VM_SIZE (Student Quota Safe)"
    echo "============================================================"

    # 1. Create Resource Group
    echo -e "\n[1/5] Creating Azure Resource Group..."
    az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

    # 2. Create AKS Cluster (1 node Standard_B2s)
    echo -e "\n[2/5] Provisioning AKS Cluster (takes ~3-4 minutes)..."
    az aks create \
      --resource-group "$RESOURCE_GROUP" \
      --name "$CLUSTER_NAME" \
      --node-count "$NODE_COUNT" \
      --node-vm-size "$VM_SIZE" \
      --generate-ssh-keys \
      --enable-managed-identity

    # 3. Get Credentials for kubectl
    echo -e "\n[3/5] Connecting kubectl to AKS..."
    az aks get-credentials --resource-group "$RESOURCE_GROUP" --name "$CLUSTER_NAME" --overwrite-existing

    # 4. Deploy Kubernetes Manifests
    echo -e "\n[4/5] Deploying API pods and LoadBalancer service..."
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml

    # 5. Wait for LoadBalancer Public IP
    echo -e "\n[5/5] Waiting for Public IP allocation from Azure LoadBalancer..."
    echo "Checking service status (may take 60-90 seconds)..."
    for i in {1..20}; do
      EXTERNAL_IP=$(kubectl get svc heart-disease-api-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)
      if [ -n "$EXTERNAL_IP" ]; then
        break
      fi
      echo "  Waiting for external IP... ($i/20)"
      sleep 6
    done

    echo "============================================================"
    if [ -n "$EXTERNAL_IP" ]; then
      echo "  SUCCESS! API is live on the internet!"
      echo "  Public URL:    http://$EXTERNAL_IP"
      echo "  Swagger Docs:  http://$EXTERNAL_IP/docs"
      echo "  Health Check:  http://$EXTERNAL_IP/health"
      echo "  Metrics:       http://$EXTERNAL_IP/metrics"
      echo "============================================================"
      echo ""
      echo "To test your live deployment run:"
      echo "  bash deploy_azure.sh test $EXTERNAL_IP"
    else
      echo "  Cluster ready, external IP is still provisioning."
      echo "  Run 'kubectl get svc heart-disease-api-service' to view IP."
    fi
    echo ""
    echo "IMPORTANT (Preserve Credits): When done testing and taking screenshots,"
    echo "run: bash deploy_azure.sh destroy"
    ;;

  status)
    echo "=== Pods ==="
    kubectl get pods -o wide
    echo -e "\n=== Services (Public IP) ==="
    kubectl get svc heart-disease-api-service
    ;;

  test)
    TARGET_IP="${2:-$(kubectl get svc heart-disease-api-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo 'localhost')}"
    echo "Testing Heart Disease API on http://$TARGET_IP..."
    curl -s "http://$TARGET_IP/health" | python3 -m json.tool
    echo ""
    curl -s -X POST "http://$TARGET_IP/predict" \
      -H "Content-Type: application/json" \
      -d '{
        "age": 54, "sex": 1, "cp": 3, "trestbps": 131, "chol": 246,
        "fbs": 0, "restecg": 1, "thalach": 149, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3
      }' | python3 -m json.tool
    ;;

  destroy)
    echo "WARNING: Deleting resource group '$RESOURCE_GROUP' will remove the AKS cluster and stop all billing."
    read -p "Are you sure? (y/N): " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
      az group delete --name "$RESOURCE_GROUP" --yes --no-wait
      echo "Resource group '$RESOURCE_GROUP' deletion initiated in background."
      echo "Billing has been stopped."
    else
      echo "Destruction cancelled."
    fi
    ;;

  *)
    echo "Usage: bash deploy_azure.sh [deploy|status|test|destroy]"
    ;;
esac
