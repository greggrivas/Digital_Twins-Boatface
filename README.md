# AIS4004 Gas Turbine Digital Twin

Last updated: 2026-02-20

This repository contains the full school project for condition-based monitoring of a marine gas turbine digital twin:

- Python FastAPI backend for predictions, maintenance logic, and HMI data endpoints
- Next.js HMI frontend with dashboard, chat assistant, and M501J 360 viewer assets
- Data analysis and model-support artifacts used for the project report

## Repository Layout

```text
.
├── Data/                    # Dataset, EDA outputs, model analysis scripts/exports
├── src/                     # FastAPI app and backend service logic
├── stitch 2/hmi-app/        # Next.js HMI app
├── openapi.json             # API schema export
├── README_API.md            # Backend endpoint reference
└── requirements.txt         # Python dependencies
```

## Quick Start

### 1) Backend (FastAPI)

```bash
cd "/Users/nicolas/Documents/Vibes/00_School/AIS4004 Digital Twins- boatface copy"
python3 -m pip install -r requirements.txt
uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

Backend docs:

- Swagger: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

### 2) Frontend (HMI)

```bash
cd "/Users/nicolas/Documents/Vibes/00_School/AIS4004 Digital Twins- boatface copy/stitch 2/hmi-app"
npm install
```

Create `.env.local` manually:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=google/gemini-3-flash-preview
FASTAPI_BASE_URL=http://127.0.0.1:8000
```

Then run:

```bash
npm run dev
```

HMI URL: `http://127.0.0.1:3000/hmi`

## Current Implemented Features

- Header branding: **HMI by IndustryStandard™** and subtitle **Gas Turbine Digital Twin**
- 360 engine viewer using local M501J frame sequence in:
  - `stitch 2/hmi-app/public/m501j360/raw`
  - `stitch 2/hmi-app/public/m501j360/clean`
  - `stitch 2/hmi-app/public/m501j360/original`
- Slider-based manual rotation, default side frame, no auto-spin
- Interactive 3D degradation surface with hover details
- TIC displayed in ship status and documented separately from Fuel_Flow
- Chat assistant with context-aware post-tool summarization (no tool-call traces in UI)

## Notes

- This repository is for academic use in AIS4004.
- HMI snapshot values are sampled from dataset rows (simulated snapshot behavior), not live ship telemetry.
