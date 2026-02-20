# Digital Twin for Marine Vessel Propulsion: Report and Implementation Structure

Last updated: 2026-02-20

## 1. Report Structure (Academic)

1. Introduction
2. Methodology
3. Digital Twin Development
4. AI Assistant Integration
5. Results and Discussion
6. Conclusion
7. References

## 2. Data and Model Artifacts in This Repository

- `Data/data.csv`: original propulsion dataset
- `Data/cleaned_data.csv`: cleaned modeling dataset used by backend services
- `Data/EDA/analysis.py`: exploratory analysis script
- `Data/EDA/*.csv`: descriptive stats and correlation outputs
- `Data/Models/models.py`: model experimentation/training script
- `Data/Models/*.csv`: model metrics and CV/regime analysis outputs
- `Data/3dview.py`: reference 3D plotting script for degradation surface studies

## 3. Current Implementation Status

### Backend (FastAPI)

- Source folder: `src/`
- Main app: `src/main.py`
- Service logic/models: `src/service.py`
- Schemas: `src/schemas.py`

Implemented endpoint groups:

- Core prediction and analysis:
  - `POST /predict/decay`
  - `POST /predict/decay-from-sensors`
  - `POST /analyze/sensors`
  - `GET /dataset/summary`
  - `GET /correlations/physical`
  - `POST /maintenance/recommend`
  - `POST /tools/compare-operating-conditions`
- HMI support:
  - `GET /hmi/snapshot`
  - `GET /hmi/surface-data`
  - `POST /hmi/surface-marker`

### Frontend (Next.js HMI)

- App folder: `stitch 2/hmi-app`
- Main shell: `components/hmi/hmi-shell.tsx`
- Dashboard: `components/hmi/turbine-dashboard.tsx`
- Chat panel: `components/hmi/chat-panel.tsx`
- M501J 360 viewer: `components/hmi/m501j-360-view.tsx`

Implemented HMI features:

- Header branding updated to IndustryStandardTM + Gas Turbine Digital Twin subtitle
- 360 turbine viewer with local frame sequence and slider control
- Interactive 3D degradation surface with hover inspection and corrected axis layout
- TIC included in ship status section
- Neutral outer styling for top recommendation card at all priorities

### Assistant Integration

- API route: `stitch 2/hmi-app/app/api/chat/route.ts`
- Context file: `stitch 2/hmi-app/data/assistant_context.md`

Current assistant behavior:

- Uses tool calls for factual data retrieval and prediction endpoints
- Applies context-aware post-tool summarization before responding
- Hides tool traces from chat UI
- Maintains deterministic fallback summary for resilience

## 4. Asset Structure for Turbine 360

Runtime and source asset folders:

- `stitch 2/hmi-app/public/m501j360/raw`
- `stitch 2/hmi-app/public/m501j360/clean`
- `stitch 2/hmi-app/public/m501j360/original`

`clean/` frames are used directly by the active HMI component.

## 5. Validation Checklist

- Backend health route responds (`GET /health`)
- HMI renders snapshot and 3D surface endpoints
- Chat assistant returns plain-language summaries without tool internals
- `.env.local` remains excluded from version control
