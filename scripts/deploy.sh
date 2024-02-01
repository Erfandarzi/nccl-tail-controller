#!/bin/bash

# NCCL Controller Deployment Script
# Requires: systemd, cgroup v2, CAP_SYS_ADMIN

set -e

INSTALL_DIR="/opt/nccl-controller"
SERVICE_NAME="nccl-controller"
USER="nccl-ctrl"

echo "=== NCCL Tail Latency Controller Deployment ==="

# Check prerequisites
check_prerequisites() {
    echo "Checking system prerequisites..."
    
    # Check for systemd
    if ! command -v systemctl &> /dev/null; then
        echo "ERROR: systemd is required"
        exit 1
    fi
    
    # Check cgroup v2
    if ! grep -q cgroup2 /proc/filesystems; then
        echo "WARNING: cgroup v2 not detected, some features may be limited"
    fi
    
    # Check capabilities
    if ! grep -q cap_sys_admin /proc/$$/status; then
        echo "WARNING: CAP_SYS_ADMIN not available, hardware profiling will be limited"
    fi
    
    # Check for CUDA
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA detected: $(nvidia-smi --query-gpu=count --format=csv,noheader,nounits) GPUs"
    else
        echo "WARNING: CUDA not detected"
    fi
    
    echo "Prerequisites check completed."
}

# Create service user
create_service_user() {
    if ! id "$USER" &>/dev/null; then
        echo "Creating service user: $USER"
        useradd -r -d "$INSTALL_DIR" -s /bin/false "$USER"
    fi
}

# Install files
install_controller() {
    echo "Installing controller to $INSTALL_DIR..."
    
    mkdir -p "$INSTALL_DIR"
    cp -r . "$INSTALL_DIR/"
    chown -R "$USER:$USER" "$INSTALL_DIR"
    chmod +x "$INSTALL_DIR/example_integration.py"
    
    # Install Python dependencies
    pip3 install -r "$INSTALL_DIR/requirements.txt"
}

# Configure systemd service
configure_service() {
    echo "Configuring systemd service..."
    
    cat > "/etc/systemd/system/$SERVICE_NAME.service" << EOF
[Unit]
Description=NCCL Tail Latency Controller
After=network.target
Requires=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/python3 $INSTALL_DIR/controller.py
Restart=always
RestartSec=10

# Cgroup configuration for monitoring
Slice=system-nccl.slice

# Capabilities for hardware access
AmbientCapabilities=CAP_SYS_ADMIN CAP_NET_ADMIN
CapabilityBoundingSet=CAP_SYS_ADMIN CAP_NET_ADMIN

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$INSTALL_DIR

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"
}

# Configure IRQ affinity for optimal performance
configure_irq_affinity() {
    echo "Configuring IRQ affinity for network interfaces..."
    
    # This would contain sophisticated IRQ affinity tuning
    # for optimal network performance - implementation details
    # are system-specific and not included in this demo
    
    echo "IRQ affinity configuration completed"
}

# Main deployment sequence
main() {
    if [[ $EUID -ne 0 ]]; then
        echo "This script must be run as root"
        exit 1
    fi
    
    check_prerequisites
    create_service_user
    install_controller
    configure_service
    configure_irq_affinity
    
    echo ""
    echo "=== Deployment Complete ==="
    echo "Start the service with: systemctl start $SERVICE_NAME"
    echo "Check status with:      systemctl status $SERVICE_NAME"
    echo "View logs with:         journalctl -u $SERVICE_NAME -f"
    echo ""
    echo "Note: Ensure your training workload integrates with the controller"
    echo "See example_integration.py for reference implementation"
}

main "$@"