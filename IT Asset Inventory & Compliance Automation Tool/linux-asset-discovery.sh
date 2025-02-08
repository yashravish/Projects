#!/bin/bash
# scripts/discovery/linux-asset-discovery.sh
serial=$(sudo dmidecode -s system-serial-number)
os=$(lsb_release -d | cut -f2)

curl -X POST https://api/asset-discovery \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "hostname": "$(hostname)",
  "os_version": "$os",
  "serial_number": "$serial",
  "install_date": "$(date -d @$(stat -c %Y /etc) '+%Y-%m-%d')"
}
EOF