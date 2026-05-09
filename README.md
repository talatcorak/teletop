# teletop

Uzaktan ESP32 flash + monitor sistemi. Raspberry Pi'ye bağlı ESP32'leri ağ üzerinden flash etmek ve seri çıktılarını canlı izlemek için.

## Mimari

- **`server/`** — Raspberry Pi üzerinde koşan FastAPI backend. ESP32 cihazlarını yönetir, flash işlemlerini yürütür, seri portu WebSocket üzerinden yayınlar.
- **`web/`** — React + Vite + Tailwind frontend. Tarayıcıdan cihaz seçimi, flash, monitor.
- **`client/`** — PC üzerinde çalışan Click tabanlı CLI. Lokal projeyi build edip RPi'ye gönderir, monitor'ü terminale stream eder.
- **`shared/`** — Bileşenler arası ortak şema/notlar.

## Kurulum

PC üzerinde geliştirme için (RPi deployment ayrı):

```bash
# Backend
cd server && uv sync

# Frontend
cd ../web && pnpm install

# CLI
cd ../client && uv sync
```

## Geliştirme

```bash
# Backend (http://localhost:8000)
cd server && uv run teletop-server

# Frontend dev server (http://localhost:5173)
cd web && pnpm dev

# CLI
cd client && uv run teletop status
```

## Konfigürasyon

Server `TELETOP_*` env vars ile yapılandırılır:

- `TELETOP_HOST` (default `0.0.0.0`)
- `TELETOP_PORT` (default `8000`)
- `TELETOP_DATA_DIR` (default `~/teletop`)
- `TELETOP_AUTH_TOKEN` (Task 12'den itibaren zorunlu)

## Breaking change in 0.2 — multi-chip support

`DeviceRegistration` artık iki ayrı chip alanı tutuyor:

- `usb_chip` — USB-UART converter (önceki `chip` alanı; CH340, CP2102, ...)
- `target_chip` — flash hedefi ESP ailesi (`esp32` / `esp8266` / `esp32s2` / `esp32s3` / `esp32c3` / `esp32c6` / `esp32h2`), **zorunlu**

0.2'den önce kaydedilmiş cihazlar `load_registry()` sırasında otomatik migrate ediliyor: `chip → usb_chip` ve eksik `target_chip = "esp32"` (uyarıyla, sonra persist). Yanlışsa düzelt:

```bash
uv run teletop-server set-target agv1 esp8266
```

Stable symlink prefix de değişti: `/dev/esp32-<alias>` → `/dev/tty-<alias>`. Yeni symlink'leri oluşturmak için:

```bash
sudo $(which uv) run teletop-server udev-install
```

`udevadm trigger` eski `/dev/esp32-*` symlink'lerini düşürmezse bir `sudo udevadm trigger --action=change` veya reboot çözer.

## Raspberry Pi initial setup

İlk kurulum için RPi'da sırayla:

```bash
# 1. Repo'yu klonla (örnek hedef: ~/teletop-src)
git clone <repo-url> ~/teletop-src
cd ~/teletop-src

# 2. uv kur (kullanıcı olarak, root değil)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. APT bağımlılıkları + dialout grup üyeliği + uv sync (idempotent)
sudo bash server/scripts/setup-rpi.sh

# 4. dialout yeni eklendiyse bir kez logout/login (oturum yenile)

# 5. Cihazları kaydet
cd server
uv run teletop-server discover                            # önce kuru tarama
uv run teletop-server register agv1                       # interaktif: target chip prompt'u sorar
# veya non-interaktif:
#   uv run teletop-server register agv1 --port 3-1 --vid 0x1A86 --pid 0x7523 --target esp8266
# auto-detect:
#   uv run teletop-server register agv1 --port 3-1 --vid 0x1A86 --pid 0x7523 --detect

# 6. udev rule'larını yükle (cihazları stable /dev/tty-<alias> olarak görünür yapar)
sudo $(which uv) run teletop-server udev-install

# 7. Doğrula
ls -l /dev/tty-*
uv run teletop-server list
```

`setup-rpi.sh` idempotent — yeniden çalıştırmak güvenli. Devices kaydetmeden çalıştırırsan udev install'ı atlar; sonradan `udev-install` ile manuel ekleyebilirsin. Yeni cihaz eklediğinde / sildiğinde CLI sana udev rule'larını yenilemeyi soracak.
