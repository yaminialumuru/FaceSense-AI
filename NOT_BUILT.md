# Remaining work
1. **Backend** (FastAPI + SQLAlchemy/Alembic + PostgreSQL/SQLite): users, password hashing, roles, session/event APIs, audit log.
2. **Authentication and role checks** enforced server-side.
3. (Removed from the UI) **Identity verification / access control**: needs a face-embedding model (e.g. an ArcFace/MobileFaceNet ONNX) chosen, license-checked and tested; enrollment with consent, embedding storage, revoke/delete, threshold, liveness (not claimed).
4. **Access policies**: decisions with reasons, admin override, audit history. Emotion-based rules stay off by default.
5. **Real-world evaluation** of yawn/drowsiness thresholds and expression accuracy with test subjects.
6. Recharts-style analytics by date range (current reports use simple bars and tables), server-side retention.
