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
LOCATION="${AZURE_LOCATION:-centralindia}"
CLUSTER_NAME="${AZURE_CLUSTER:-aks-heart-mlops}"
VM_SIZE="Standard_B2s_v2"   # 2 vCPUs - strictly safe for student account quota in centralindia
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
    echo -e "\n[1/6] Creating Azure Resource Group..."
    az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

    # 2. Find or Create Azure Container Registry (ACR)
    EXISTING_ACR=$(az acr list --resource-group "$RESOURCE_GROUP" --query "[0].name" -o tsv 2>/dev/null || true)
    if [ -n "$EXISTING_ACR" ] && [ "$EXISTING_ACR" != "None" ]; then
      ACR_NAME="$EXISTING_ACR"
      echo -e "\n[2/6] Using existing Azure Container Registry ($ACR_NAME)..."
    else
      RAND_SUFFIX=$(openssl rand -hex 3 2>/dev/null || echo "101")
      ACR_NAME="${AZURE_ACR:-acrheart$RAND_SUFFIX}"
      echo -e "\n[2/6] Provisioning Azure Container Registry ($ACR_NAME)..."
      az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic --admin-enabled true
    fi
    ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer --output tsv)

    # 3. Build & push container image for linux/amd64 natively compatible with AKS VMs
    IMAGE_EXISTS=$(az acr repository show-tags --name "$ACR_NAME" --repository heart-disease-api --query "[?@=='latest']" -o tsv 2>/dev/null || true)
    if [ -n "$IMAGE_EXISTS" ]; then
      echo -e "\n[3/6] Container image already exists in ACR ($ACR_LOGIN_SERVER/heart-disease-api:latest). Skipping build!"
    else
      echo -e "\n[3/6] Authenticating and pushing linux/amd64 container image to ACR ($ACR_LOGIN_SERVER)..."
      az acr login --name "$ACR_NAME"
      docker buildx build --platform linux/amd64 -t "$ACR_LOGIN_SERVER/heart-disease-api:latest" --push .
    fi

    # 4. Create AKS Cluster (1 node Standard_B2s, attached to ACR)
    echo -e "\n[4/6] Provisioning AKS Cluster (takes ~3-4 minutes)..."
    az aks create \
      --resource-group "$RESOURCE_GROUP" \
      --name "$CLUSTER_NAME" \
      --node-count "$NODE_COUNT" \
      --node-vm-size "$VM_SIZE" \
      --generate-ssh-keys \
      --attach-acr "$ACR_NAME"

    # 5. Get Credentials for kubectl
    echo -e "\n[5/6] Connecting kubectl to AKS..."
    az aks get-credentials --resource-group "$RESOURCE_GROUP" --name "$CLUSTER_NAME" --overwrite-existing

    # 6. Deploy Kubernetes Manifests
    echo -e "\n[6/6] Deploying API pods and LoadBalancer service..."
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/hpa.yaml 2>/dev/null || true
    
    # Deploy with ACR image
    sed "s|image: heart-disease-api:latest|image: $ACR_LOGIN_SERVER/heart-disease-api:latest|g; s|imagePullPolicy: IfNotPresent|imagePullPolicy: Always|g" k8s/deployment.yaml | kubectl apply -f -

    # 7. Wait for LoadBalancer Public IP
    echo -e "\n[7/7] Waiting for Public IP allocation from Azure LoadBalancer..."
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
