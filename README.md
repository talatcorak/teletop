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
uv run teletop-server discover                 # önce kuru tarama
uv run teletop-server register agv1            # interaktif (önerilen)
# veya: uv run teletop-server register agv1 --port 3-1 --vid 0x1A86 --pid 0x7523

# 6. udev rule'larını yükle (cihazları stable /dev/esp32-<alias> olarak görünür yapar)
sudo $(which uv) run teletop-server udev-install

# 7. Doğrula
ls -l /dev/esp32-*
uv run teletop-server list
```

`setup-rpi.sh` idempotent — yeniden çalıştırmak güvenli. Devices kaydetmeden çalıştırırsan udev install'ı atlar; sonradan `udev-install` ile manuel ekleyebilirsin. Yeni cihaz eklediğinde / sildiğinde CLI sana udev rule'larını yenilemeyi soracak.
