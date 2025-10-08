# NeuroDrive DMS - Full Project (Backend + Frontend)

Production-ready scaffold for a real-time Driver Monitoring System dashboard with:

- Backend: FastAPI service exposing inference and config
- Model: placeholder at `backend/model/file.py`
- Config: `config.yaml` with weights and thresholds
- Frontend: single-file React dashboard `NeuroDriveDashboard.jsx` using Firestore + simulation

Additionally, a Java Spring Boot service is provided for enterprise orchestration, JWT/RBAC placeholders, and endpoints suitable for admin dashboards, SOS, and family sharing integrations.

## Prerequisites

- Python 3.10+
- Node.js 18+ (if bundling frontend)
- Firebase project (if using Firestore)

## Backend (FastAPI)

Install deps:

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run API:

```bash
set NEURODRIVE_CONFIG=%CD%\config.yaml  # PowerShell: $env:NEURODRIVE_CONFIG = (Get-Location).Path + "\config.yaml"
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:

- GET `/health` – service status
- GET `/config` – current weights/thresholds
- POST `/api/v1/infer` – compute risk from metrics

Example request:

```bash
curl -X POST http://localhost:8000/api/v1/infer \
 -H "Content-Type: application/json" \
 -d '{"eyeClosureRatio":0.3,"phoneUsage":true,"speed":92}'
```

## Frontend (React single file)

Use `NeuroDriveDashboard.jsx` directly inside your React app. Ensure `firebase` packages are installed and Tailwind CSS is present.

Example mount:

```jsx
import NeuroDriveDashboard from './NeuroDriveDashboard.jsx';

<NeuroDriveDashboard
  firebaseConfig={{ /* from Firebase Console */ }}
  appId="neurodrive-prod"
  __initial_auth_token={window.__initial_auth_token}
/>;
```

Optional global config:

```js
window.NEURODRIVE_CONFIG = {
  firebaseConfig: {/* ... */},
  appId: 'neurodrive-prod',
  __initial_auth_token: '<custom_jwt>',
  riskWeights: { drowsiness: 0.25, distraction: 0.30, headPose: 0.25, unauthorizedObjects: 0.20 },
  thresholds: { eyeClosureTimeSeconds: 2.0, distractionDetectedThreshold: 0.5, headPoseDegreesThreshold: 15, unauthorizedObjectsThreshold: 1, speedLimitKmh: 80 }
};
```

## AI Model File

- Located at `backend/model/file.py`
- Replace stub with real model loading (e.g., ONNX/Torch) and inference routines.

## Connecting Frontend to Backend

- The dashboard simulates and writes metrics to Firestore. To integrate backend inference, have the dashboard call `POST /api/v1/infer` with current control values and write the response to Firestore or local state instead of its internal calculator.

## Security & Production Notes

- Lock down Firestore rules; avoid wide-open writes in production.
- Replace anonymous auth with verified custom tokens.
- Add request validation and rate limiting server-side.
- Containerize FastAPI; deploy with HTTPS termination.

## Spring Boot Service

Build and run:

```bash
cd springboot
mvn spring-boot:run
```

### Endpoints

**Authentication:**
- POST `/api/v1/auth/register` – Register new user
- POST `/api/v1/auth/login` – Login with JWT token

**Core Services:**
- GET `/api/v1/health` – Service health check
- POST `/api/v1/infer` – Risk calculation (mirrors FastAPI)

**SOS & Emergency:**
- POST `/api/v1/sos/activate` – Activate SOS alert
- POST `/api/v1/sos/deactivate` – Deactivate SOS

**Family Sharing:**
- POST `/api/v1/family/members` – Add family member
- GET `/api/v1/family/members` – List family members
- GET `/api/v1/family/driver-status/{driverId}` – Get driver status

**WebSocket:**
- `/ws` – WebSocket endpoint for real-time notifications
- Topics: `/topic/sos/{familyMemberId}`, `/topic/alerts/{familyMemberId}`

### Database Setup

Configure DBs via `docker-compose.yml`:

```bash
docker compose up -d
```

This starts:
- PostgreSQL (port 5432) – Structured data
- MongoDB (port 27017) – Unstructured logs  
- Redis (port 6379) – Caching

Spring Boot `application.properties` is configured to connect to these services automatically.

### Security

- JWT-based authentication with role-based access control (RBAC)
- Roles: `DRIVER`, `ADMIN`, `FAMILY_MEMBER`
- Password encryption with BCrypt
- Secure endpoints with proper authorization

## Usage Examples

### Register and Login

```bash
# Register new user
curl -X POST http://localhost:8080/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "driver1",
    "email": "driver@example.com",
    "password": "password123",
    "firstName": "John",
    "lastName": "Doe"
  }'

# Login
curl -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "driver1",
    "password": "password123"
  }'
```

### SOS Activation

```bash
# Activate SOS (requires JWT token)
curl -X POST http://localhost:8080/api/v1/sos/activate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "location": "Highway 101, Mile 45",
    "latitude": 37.7749,
    "longitude": -122.4194,
    "reason": "Medical emergency"
  }'
```

### Family Member Management

```bash
# Add family member
curl -X POST http://localhost:8080/api/v1/family/members \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "name": "Jane Doe",
    "phoneNumber": "+1234567890",
    "email": "jane@example.com",
    "relationship": "spouse",
    "canReceiveAlerts": true,
    "canViewLocation": true
  }'
```

## Integration Guide

### Frontend-Backend Integration

The React dashboard can be configured to use either FastAPI or Spring Boot for inference:

```jsx
// Use FastAPI backend
<NeuroDriveDashboard
  firebaseConfig={firebaseConfig}
  appId="neurodrive-prod"
  backendUrl="http://localhost:8000/api/v1/infer"
/>

// Use Spring Boot backend
<NeuroDriveDashboard
  firebaseConfig={firebaseConfig}
  appId="neurodrive-prod"
  backendUrl="http://localhost:8080/api/v1/infer"
/>
```

### WebSocket Integration

Connect to real-time notifications:

```javascript
import SockJS from 'sockjs-client';
import Stomp from 'stompjs';

const socket = new SockJS('http://localhost:8080/ws');
const stompClient = Stomp.over(socket);

stompClient.connect({}, () => {
  // Subscribe to SOS alerts
  stompClient.subscribe('/topic/sos/1', (message) => {
    const alert = JSON.parse(message.body);
    console.log('SOS Alert:', alert);
  });
  
  // Subscribe to risk alerts
  stompClient.subscribe('/topic/alerts/1', (message) => {
    const alert = JSON.parse(message.body);
    console.log('Risk Alert:', alert);
  });
});
```

## Production Deployment

### Docker Deployment

Create `Dockerfile` for each service:

```dockerfile
# FastAPI Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# Spring Boot Dockerfile
FROM openjdk:21-jdk-slim
WORKDIR /app
COPY springboot/target/neurodrive-backend-1.0.0.jar app.jar
EXPOSE 8080
CMD ["java", "-jar", "app.jar"]
```

### Kubernetes Deployment

Deploy with `kubectl`:

```bash
# Apply database services
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/mongodb.yaml
kubectl apply -f k8s/redis.yaml

# Apply application services
kubectl apply -f k8s/fastapi.yaml
kubectl apply -f k8s/springboot.yaml
```

## AI Model Customization

The AI model in `backend/model/file.py` can be extended with:

- Real facial landmark detection (dlib, MediaPipe)
- Emotion recognition models (FER2013, AffectNet)
- Object detection for phone usage (YOLO, SSD)
- Audio processing for distraction detection (Librosa, PyAudio)

Replace the placeholder implementations with your trained models for production use.

## Monitoring and Logging

- Spring Boot Actuator endpoints: `/actuator/health`, `/actuator/metrics`
- FastAPI health check: `/health`
- Application logs with configurable levels
- Database connection monitoring
- WebSocket connection status



