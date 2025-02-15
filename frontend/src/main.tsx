import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';
import { CurrConfigProvider } from './context.tsx'; // Import the provider

createRoot(document.getElementById('root')!).render(
  <CurrConfigProvider> 
    <StrictMode>
      <App />
    </StrictMode>
  </CurrConfigProvider>
);
