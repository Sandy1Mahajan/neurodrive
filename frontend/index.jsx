// Entry point for React frontend
import React from 'react';
import ReactDOM from 'react-dom/client';
import NeuroDriveDashboard from '../NeuroDriveDashboard.jsx';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <NeuroDriveDashboard />
  </React.StrictMode>
);
