#!/bin/bash
# Klipper Automatic Watchdog & Non-Critical Error Handler
LOG_FILE="/home/shoomba/printer_data/logs/klippy.log"
WATCHDOG_LOG="/home/shoomba/printer_data/logs/watchdog.log"

echo "$(date): Klipper Watchdog iniciado." >> "$WATCHDOG_LOG"

tail -Fn0 "$LOG_FILE" | while read -r line; do
    if echo "$line" | grep -Ei "Timer too close|Lost communication with MCU|MCU .* shutdown|out of range"; then
        echo "$(date): Erro não crítico/desligamento detectado: $line" >> "$WATCHDOG_LOG"
        
        # Aguarda 3 segundos para estabilização
        sleep 3
        
        # Tenta reiniciar o serviço Klipper suavemente
        systemctl restart klipper.service
        echo "$(date): Reinício do Klipper acionado automaticamente." >> "$WATCHDOG_LOG"
    fi
done
