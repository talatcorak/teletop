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
