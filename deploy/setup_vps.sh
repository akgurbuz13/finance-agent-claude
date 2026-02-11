#!/usr/bin/env bash
set -euo pipefail

# Portfolio Advisor VPS Setup Script
# Run as root on a fresh Ubuntu 22.04+ VPS

APP_DIR="/opt/portfolio-advisor"
APP_USER="portfolio"

echo "=== Portfolio Advisor VPS Setup ==="

# 1. System updates
echo ">> Updating system..."
apt-get update -y && apt-get upgrade -y

# 2. Install Python 3.11+
echo ">> Installing Python..."
apt-get install -y python3.11 python3.11-venv python3-pip git

# 3. Create app user
echo ">> Creating app user..."
if ! id "$APP_USER" &>/dev/null; then
    useradd --system --create-home --shell /bin/bash "$APP_USER"
fi

# 4. Clone/setup app directory
echo ">> Setting up app directory..."
mkdir -p "$APP_DIR/data"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# 5. Setup virtual environment
echo ">> Creating virtual environment..."
sudo -u "$APP_USER" python3.11 -m venv "$APP_DIR/.venv"

# 6. Install dependencies
echo ">> Installing dependencies..."
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -e "$APP_DIR"

# 7. Setup .env file
if [ ! -f "$APP_DIR/.env" ]; then
    echo ">> Creating .env template..."
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    chmod 600 "$APP_DIR/.env"
    chown "$APP_USER:$APP_USER" "$APP_DIR/.env"
    echo "!! IMPORTANT: Edit $APP_DIR/.env with your API keys !!"
fi

# 8. Install systemd service
echo ">> Installing systemd service..."
cp "$APP_DIR/deploy/portfolio-advisor.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable portfolio-advisor

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit $APP_DIR/.env with your API keys"
echo "  2. Start the service: systemctl start portfolio-advisor"
echo "  3. Check status: systemctl status portfolio-advisor"
echo "  4. View logs: journalctl -u portfolio-advisor -f"
