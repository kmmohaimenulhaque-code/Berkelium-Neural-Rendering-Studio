// Basic renderer entry point - this will be expanded in later phases

import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from '../src/App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);