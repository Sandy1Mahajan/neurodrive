import React, { useEffect, useMemo, useRef, useState } from 'react';

// Single-file NeuroDrive DMS Web Dashboard
// - Frontend: React (functional components, hooks)
// - Styling: Tailwind CSS class names (assumes Tailwind available in host app)
// - Backend/Data: Firestore realtime doc + in-app simulation writer
// - Auth: Custom token via global __initial_auth_token or anonymous fallback

// Firebase modular SDK imports (v9+). Ensure firebase packages are installed in host app.
// If using via <script> tags, adjust to compat namespace accordingly.
import { initializeApp, getApps } from 'firebase/app';
import {
  getAuth,
  onAuthStateChanged,
  signInWithCustomToken,
  signInAnonymously,
} from 'firebase/auth';
import {
  getFirestore,
  doc,
  onSnapshot,
  setDoc,
  serverTimestamp,
} from 'firebase/firestore';

// Types
/**
 * @typedef {Object} RiskWeights
 * @property {number} drowsiness
 * @property {number} distraction
 * @property {number} headPose
 * @property {number} unauthorizedObjects
 */

/**
 * @typedef {Object} Thresholds
 * @property {number} eyeClosureTimeSeconds
 * @property {number} distractionDetectedThreshold
 * @property {number} headPoseDegreesThreshold
 * @property {number} unauthorizedObjectsThreshold
 * @property {number} speedLimitKmh
 */

// Default config (used if not provided via props or global config)
const DEFAULT_RISK_WEIGHTS = {
  drowsiness: 0.25,
  distraction: 0.30,
  headPose: 0.25,
  unauthorizedObjects: 0.20,
};

const DEFAULT_THRESHOLDS = {
  eyeClosureTimeSeconds: 2.0, // from config.yaml prompt
  distractionDetectedThreshold: 0.5, // boolean-ish threshold
  headPoseDegreesThreshold: 15, // degrees off-center
  unauthorizedObjectsThreshold: 1, // count-based
  speedLimitKmh: 80,
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function computeStatusText(score) {
  if (score < 30) return 'Critical Risk';
  if (score < 50) return 'High Risk';
  if (score < 70) return 'Elevated Risk';
  if (score < 85) return 'Low Risk';
  return 'Alert';
}

function scoreToColor(score) {
  if (score < 30) return 'text-red-500';
  if (score < 70) return 'text-yellow-500';
  return 'text-green-500';
}

function ringColor(score) {
  if (score < 30) return 'stroke-red-500';
  if (score < 70) return 'stroke-yellow-400';
  return 'stroke-green-500';
}

function metricStatus(ok) {
  return ok ? { text: 'OK', color: 'text-green-600', bg: 'bg-green-50' } : { text: 'ALERT', color: 'text-red-600', bg: 'bg-red-50' };
}

function seededRandom(seedRef) {
  // Deterministic-ish but simple: xorshift-like
  let x = seedRef.current || 123456789;
  x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
  seedRef.current = x >>> 0;
  return (seedRef.current % 10000) / 10000;
}

function Gauge({ value }) {
  const radius = 80;
  const stroke = 14;
  const normalizedRadius = radius - stroke;
  const circumference = normalizedRadius * 2 * Math.PI;
  const clamped = clamp(value, 0, 100);
  const strokeDashoffset = circumference - (clamped / 100) * circumference;
  const ringCls = ringColor(clamped);
  return (
    <svg height={radius * 2} width={radius * 2} className="overflow-visible">
      <circle
        stroke="#e5e7eb"
        fill="transparent"
        strokeWidth={stroke}
        r={normalizedRadius}
        cx={radius}
        cy={radius}
      />
      <circle
        strokeLinecap="round"
        className={ringCls}
        fill="transparent"
        strokeWidth={stroke}
        r={normalizedRadius}
        cx={radius}
        cy={radius}
        style={{ strokeDasharray: `${circumference} ${circumference}`, strokeDashoffset }}
        transform={`rotate(-90 ${radius} ${radius})`}
      />
      <text x="50%" y="50%" dominantBaseline="middle" textAnchor="middle" className={`font-semibold text-3xl ${scoreToColor(clamped)}`}>
        {Math.round(clamped)}
      </text>
    </svg>
  );
}

function MetricCard({ title, value, threshold, ok, unit }) {
  const st = metricStatus(ok);
  return (
    <div className={`rounded-xl border border-gray-200 p-4 ${st.bg}`}>
      <div className="flex items-center justify-between">
        <div className="text-gray-600 text-sm font-medium">{title}</div>
        <div className={`text-xs px-2 py-0.5 rounded-full ${ok ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>{st.text}</div>
      </div>
      <div className="mt-3 flex items-end gap-2">
        <div className="text-2xl font-semibold text-gray-900">{value}{unit ? <span className="text-sm text-gray-500 ml-1">{unit}</span> : null}</div>
      </div>
      <div className="mt-1 text-xs text-gray-500">Threshold: {threshold}{unit ? ` ${unit}` : ''}</div>
    </div>
  );
}

function EventLog({ events }) {
  return (
    <div className="h-56 overflow-y-auto rounded-xl border border-gray-200 bg-white">
      <div className="sticky top-0 bg-gray-50 px-4 py-2 text-xs font-semibold text-gray-600 border-b">Real-Time Alerts & Events</div>
      <ul className="divide-y">
        {events.length === 0 ? (
          <li className="px-4 py-3 text-sm text-gray-500">No events yet.</li>
        ) : events.map((e, idx) => (
          <li key={idx} className="px-4 py-3 text-sm">
            <div className="flex items-center justify-between">
              <span className="font-medium text-gray-800">{e.type}</span>
              <span className="text-xs text-gray-500">{new Date(e.ts).toLocaleTimeString()}</span>
            </div>
            {e.detail ? <div className="text-gray-600 text-xs mt-0.5">{e.detail}</div> : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function NeuroDriveDashboard({
  firebaseConfig,
  appId,
  __initial_auth_token,
  riskWeights: riskWeightsProp,
  thresholds: thresholdsProp,
  backendUrl,
}) {
  const [app, setApp] = useState(null);
  const [authUser, setAuthUser] = useState(null);
  const [db, setDb] = useState(null);
  const [statusDocData, setStatusDocData] = useState(null);
  const [events, setEvents] = useState([]);

  // Simulation controls
  const [eyeClosureRatio, setEyeClosureRatio] = useState(0.2); // 0.0 - 1.0
  const [phoneUsage, setPhoneUsage] = useState(false);
  const [speed, setSpeed] = useState(65);

  const rngSeed = useRef(987654321);
  const writeTimerRef = useRef(null);

  const cfg = useMemo(() => {
    // Allow passing via props or global window.NEURODRIVE_CONFIG
    const globalCfg = typeof window !== 'undefined' ? window.NEURODRIVE_CONFIG : undefined;
    return {
      firebaseConfig: firebaseConfig || (globalCfg && globalCfg.firebaseConfig) || null,
      appId: appId || (globalCfg && globalCfg.appId) || 'demo-app',
      token: __initial_auth_token || (globalCfg && globalCfg.__initial_auth_token) || null,
      riskWeights: riskWeightsProp || (globalCfg && globalCfg.riskWeights) || DEFAULT_RISK_WEIGHTS,
      thresholds: thresholdsProp || (globalCfg && globalCfg.thresholds) || DEFAULT_THRESHOLDS,
      backendUrl: backendUrl || (globalCfg && globalCfg.backendUrl) || null,
    };
  }, [firebaseConfig, appId, __initial_auth_token, riskWeightsProp, thresholdsProp, backendUrl]);

  const docRefDescriptor = useMemo(() => {
    const path = `/artifacts/${cfg.appId}/public/data/neurodrive_status/current_state`;
    const parts = path.split('/').filter(Boolean);
    return { path, parts };
  }, [cfg.appId]);

  // Initialize Firebase app
  useEffect(() => {
    if (!cfg.firebaseConfig) return;
    const existing = getApps();
    const appInstance = existing.length ? existing[0] : initializeApp(cfg.firebaseConfig);
    setApp(appInstance);
    setDb(getFirestore(appInstance));
    const auth = getAuth(appInstance);
    const unsub = onAuthStateChanged(auth, (user) => {
      setAuthUser(user);
    });
    // Auth bootstrap
    (async () => {
      try {
        if (cfg.token) {
          await signInWithCustomToken(auth, cfg.token);
        } else if (!auth.currentUser) {
          await signInAnonymously(auth);
        }
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error('Auth error', err);
      }
    })();
    return () => unsub();
  }, [cfg.firebaseConfig, cfg.token]);

  // Subscribe to Firestore doc
  useEffect(() => {
    if (!db) return;
    const d = doc(db, ...docRefDescriptor.parts);
    const unsub = onSnapshot(d, (snap) => {
      const data = snap.data();
      setStatusDocData(data || null);
    });
    return () => unsub();
  }, [db, docRefDescriptor]);

  // Backend inference integration: call FastAPI or Spring Boot, then write to Firestore
  useEffect(() => {
    if (!db || !authUser) return;
    const d = doc(db, ...docRefDescriptor.parts);
    if (writeTimerRef.current) clearInterval(writeTimerRef.current);

    writeTimerRef.current = setInterval(async () => {
      const now = Date.now();
      const weights = cfg.riskWeights;
      const th = cfg.thresholds;

      // Call backend inference API
      let backendResult = null;
      try {
        const backendUrl = cfg.backendUrl || 'http://localhost:8000/api/v1/infer';
        const response = await fetch(backendUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            eyeClosureRatio: eyeClosureRatio,
            phoneUsage: phoneUsage,
            speed: speed
          })
        });
        
        if (response.ok) {
          backendResult = await response.json();
        }
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn('Backend inference failed, using local calculation', err);
      }

      // Fallback to local calculation if backend unavailable
      let riskScore, statusText, metrics, newEvents;
      
      if (backendResult && backendResult.riskScore !== undefined) {
        riskScore = backendResult.riskScore;
        statusText = backendResult.statusText || computeStatusText(riskScore);
        
        // Use backend metrics if available, otherwise generate locally
        if (backendResult.metrics) {
          metrics = backendResult.metrics;
          newEvents = backendResult.alerts || [];
        } else {
          // Generate metrics from controls for consistency
          const eyeClosureTimeSeconds = eyeClosureRatio * 3.0;
          const distractionDetected = phoneUsage;
          const headPoseDegrees = 5.0; // default
          const unauthorizedObjectsCount = 0;
          
          const drowsinessOk = eyeClosureTimeSeconds < th.eyeClosureTimeSeconds;
          const distractionOk = !distractionDetected;
          const headPoseOk = headPoseDegrees <= th.headPoseDegreesThreshold;
          const objectsOk = unauthorizedObjectsCount <= th.unauthorizedObjectsThreshold;
          const speedOk = speed <= th.speedLimitKmh;

          metrics = {
            drowsiness: { value: Number(eyeClosureTimeSeconds.toFixed(2)), threshold: th.eyeClosureTimeSeconds, unit: 's', ok: drowsinessOk },
            distraction: { value: distractionDetected ? 1 : 0, threshold: th.distractionDetectedThreshold, unit: '', ok: distractionOk },
            headPose: { value: Number(headPoseDegrees.toFixed(1)), threshold: th.headPoseDegreesThreshold, unit: '°', ok: headPoseOk },
            unauthorizedObjects: { value: unauthorizedObjectsCount, threshold: th.unauthorizedObjectsThreshold, unit: '', ok: objectsOk },
            speed: { value: speed, threshold: th.speedLimitKmh, unit: 'km/h', ok: speedOk },
          };

          newEvents = [];
          if (!drowsinessOk) newEvents.push({ type: 'Microsleep Event', detail: `Eye closure ${eyeClosureTimeSeconds.toFixed(2)}s`, ts: now });
          if (!distractionOk) newEvents.push({ type: 'Distraction Detected', detail: phoneUsage ? 'Phone usage' : 'Gaze away', ts: now });
          if (!headPoseOk) newEvents.push({ type: 'Head Pose Deviation', detail: `${headPoseDegrees.toFixed(1)}°`, ts: now });
          if (!objectsOk) newEvents.push({ type: 'Unauthorized Object', detail: `Count ${unauthorizedObjectsCount}`, ts: now });
          if (!speedOk) newEvents.push({ type: 'Speed Violation', detail: `${speed} km/h > ${th.speedLimitKmh} km/h`, ts: now });
        }
      } else {
        // Full local calculation fallback
        const eyeClosureTimeSeconds = eyeClosureRatio * 3.0 + seededRandom(rngSeed) * 0.3;
        const distractionDetected = phoneUsage || seededRandom(rngSeed) > 0.92;
        const headPoseDegrees = Math.abs((seededRandom(rngSeed) - 0.5) * 40);
        const unauthorizedObjectsCount = Math.round((seededRandom(rngSeed) > 0.97) ? 1 : 0);

        const drowsinessOk = eyeClosureTimeSeconds < th.eyeClosureTimeSeconds;
        const distractionOk = !distractionDetected;
        const headPoseOk = headPoseDegrees <= th.headPoseDegreesThreshold;
        const objectsOk = unauthorizedObjectsCount <= th.unauthorizedObjectsThreshold;
        const speedOk = speed <= th.speedLimitKmh;

        const drowsinessSub = drowsinessOk ? 100 : clamp(100 - ((eyeClosureTimeSeconds - th.eyeClosureTimeSeconds) * 40), 0, 100);
        const distractionSub = distractionOk ? 100 : 35;
        const headPoseSub = headPoseOk ? 100 : clamp(100 - ((headPoseDegrees - th.headPoseDegreesThreshold) * 3), 0, 100);
        const objectsSub = objectsOk ? 100 : 50;
        const speedSub = speedOk ? 100 : clamp(100 - ((speed - th.speedLimitKmh) * 1.2), 0, 100);

        const baseWeighted = (
          drowsinessSub * weights.drowsiness +
          distractionSub * weights.distraction +
          headPoseSub * weights.headPose +
          objectsSub * weights.unauthorizedObjects
        ) / (weights.drowsiness + weights.distraction + weights.headPose + weights.unauthorizedObjects);

        riskScore = clamp(Math.round(baseWeighted * (speedSub / 100)), 0, 100);
        statusText = computeStatusText(riskScore);

        metrics = {
          drowsiness: { value: Number(eyeClosureTimeSeconds.toFixed(2)), threshold: th.eyeClosureTimeSeconds, unit: 's', ok: drowsinessOk },
          distraction: { value: distractionDetected ? 1 : 0, threshold: th.distractionDetectedThreshold, unit: '', ok: distractionOk },
          headPose: { value: Number(headPoseDegrees.toFixed(1)), threshold: th.headPoseDegreesThreshold, unit: '°', ok: headPoseOk },
          unauthorizedObjects: { value: unauthorizedObjectsCount, threshold: th.unauthorizedObjectsThreshold, unit: '', ok: objectsOk },
          speed: { value: speed, threshold: th.speedLimitKmh, unit: 'km/h', ok: speedOk },
        };

        newEvents = [];
        if (!drowsinessOk) newEvents.push({ type: 'Microsleep Event', detail: `Eye closure ${eyeClosureTimeSeconds.toFixed(2)}s`, ts: now });
        if (!distractionOk) newEvents.push({ type: 'Distraction Detected', detail: phoneUsage ? 'Phone usage' : 'Gaze away', ts: now });
        if (!headPoseOk) newEvents.push({ type: 'Head Pose Deviation', detail: `${headPoseDegrees.toFixed(1)}°`, ts: now });
        if (!objectsOk) newEvents.push({ type: 'Unauthorized Object', detail: `Count ${unauthorizedObjectsCount}`, ts: now });
        if (!speedOk) newEvents.push({ type: 'Speed Violation', detail: `${speed} km/h > ${th.speedLimitKmh} km/h`, ts: now });
      }

      if (newEvents.length) {
        setEvents((prev) => {
          const merged = [...newEvents, ...prev];
          return merged.slice(0, 100);
        });
      }

      const payload = {
        appId: cfg.appId,
        userId: authUser && authUser.uid ? authUser.uid : 'anonymous',
        riskScore,
        statusText,
        metrics,
        weights: cfg.riskWeights,
        thresholds: cfg.thresholds,
        alerts: newEvents,
        backendResult: backendResult ? 'backend' : 'local',
        updatedAt: now,
        updatedAtServer: serverTimestamp(),
      };

      try {
        await setDoc(d, payload, { merge: true });
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error('Firestore write error', err);
      }
    }, 1200);

    return () => {
      if (writeTimerRef.current) clearInterval(writeTimerRef.current);
    };
  }, [db, authUser, eyeClosureRatio, phoneUsage, speed, cfg.riskWeights, cfg.thresholds, cfg.backendUrl, docRefDescriptor]);

  const riskScore = statusDocData && typeof statusDocData.riskScore === 'number' ? statusDocData.riskScore : 100;
  const statusText = statusDocData && statusDocData.statusText ? statusDocData.statusText : 'Initializing';

  const metrics = statusDocData && statusDocData.metrics ? statusDocData.metrics : {
    drowsiness: { value: 0, threshold: cfg.thresholds.eyeClosureTimeSeconds, unit: 's', ok: true },
    distraction: { value: 0, threshold: cfg.thresholds.distractionDetectedThreshold, unit: '', ok: true },
    headPose: { value: 0, threshold: cfg.thresholds.headPoseDegreesThreshold, unit: '°', ok: true },
    unauthorizedObjects: { value: 0, threshold: cfg.thresholds.unauthorizedObjectsThreshold, unit: '', ok: true },
    speed: { value: speed, threshold: cfg.thresholds.speedLimitKmh, unit: 'km/h', ok: true },
  };

  const userId = statusDocData && statusDocData.userId ? statusDocData.userId : (authUser ? authUser.uid : 'anonymous');

  return (
    <div className="min-h-screen w-full bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">NeuroDrive DMS Dashboard</h1>
            <p className="text-sm text-gray-600">App ID: <span className="font-mono">{cfg.appId}</span> · User: <span className="font-mono">{userId}</span></p>
          </div>
          <div className="flex items-center gap-3">
            <span className={`text-sm font-semibold ${scoreToColor(riskScore)}`}>{statusText}</span>
          </div>
        </header>

        <div className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="col-span-1 lg:col-span-1">
            <div className="rounded-2xl bg-white border border-gray-200 p-6 flex flex-col items-center">
              <div className="text-sm font-medium text-gray-600 mb-2">Unified Risk Score</div>
              <Gauge value={riskScore} />
              <div className="mt-4 text-sm text-gray-500">Green &gt; 70 · Yellow 30–70 · Red &lt; 30</div>
            </div>
          </div>

          <div className="col-span-1 lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
            <MetricCard title="Drowsiness" value={metrics.drowsiness.value} threshold={metrics.drowsiness.threshold} ok={metrics.drowsiness.ok} unit={metrics.drowsiness.unit} />
            <MetricCard title="Distraction" value={metrics.distraction.value} threshold={metrics.distraction.threshold} ok={metrics.distraction.ok} unit={metrics.distraction.unit} />
            <MetricCard title="Head Pose" value={metrics.headPose.value} threshold={metrics.headPose.threshold} ok={metrics.headPose.ok} unit={metrics.headPose.unit} />
            <MetricCard title="Unauthorized Objects" value={metrics.unauthorizedObjects.value} threshold={metrics.unauthorizedObjects.threshold} ok={metrics.unauthorizedObjects.ok} unit={metrics.unauthorizedObjects.unit} />
          </div>
        </div>

        <div className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="col-span-1 lg:col-span-2">
            <EventLog events={events} />
          </div>
          <div className="col-span-1">
            <div className="rounded-2xl bg-white border border-gray-200 p-6">
              <div className="text-sm font-semibold text-gray-700 mb-4">Simulation Controls</div>
              <div className="space-y-5">
                <div>
                  <div className="flex items-center justify-between text-sm text-gray-700">
                    <label className="font-medium">Eye Closure Ratio</label>
                    <span className="font-mono">{eyeClosureRatio.toFixed(2)}</span>
                  </div>
                  <input type="range" min={0} max={1} step={0.01} value={eyeClosureRatio} onChange={(e) => setEyeClosureRatio(parseFloat(e.target.value))} className="w-full mt-2" />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm font-medium text-gray-700">Phone Usage Detected</div>
                    <div className="text-xs text-gray-500">Toggles distraction</div>
                  </div>
                  <button onClick={() => setPhoneUsage((v) => !v)} className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${phoneUsage ? 'bg-red-500' : 'bg-gray-300'}`}>
                    <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${phoneUsage ? 'translate-x-6' : 'translate-x-1'}`} />
                  </button>
                </div>
                <div>
                  <div className="flex items-center justify-between text-sm text-gray-700">
                    <label className="font-medium">Speed (km/h)</label>
                    <span className="font-mono">{speed}</span>
                  </div>
                  <input type="number" min={0} max={200} step={1} value={speed} onChange={(e) => setSpeed(parseInt(e.target.value || '0', 10))} className="mt-2 w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500" />
                  <div className="text-xs text-gray-500 mt-1">Limit: {cfg.thresholds.speedLimitKmh} km/h</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <footer className="mt-10 text-center text-xs text-gray-500">
          Data path: <span className="font-mono">{docRefDescriptor.path}</span>
        </footer>
      </div>
    </div>
  );
}


