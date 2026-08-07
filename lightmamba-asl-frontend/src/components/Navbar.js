import React from 'react';
import { useLocation } from 'react-router-dom';
import { useBackendStatus } from '../hooks/useBackendStatus';

const ROUTE_LABELS = {
  '/':          'Dashboard',
  '/recognize': 'Video Recognition',
  '/live':      'Live Recognition',
  '/skeleton':  'Skeleton Tracking',
  '/model':     'Model Architecture',
  '/results':   'Research Results',
  '/about':     'About Project',
};

const HamburgerIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/>
  </svg>
);

const ChevronRight = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="9 18 15 12 9 6"/>
  </svg>
);

export default function Navbar({ onMenuToggle }) {
  const location = useLocation();
  const { online } = useBackendStatus();
  const pageLabel = ROUTE_LABELS[location.pathname] || 'LightMamba-ASL';

  return (
    <header className="navbar">
      <div className="navbar-left">
        <button className="navbar-hamburger" onClick={onMenuToggle} aria-label="Toggle menu">
          <HamburgerIcon />
        </button>
        <div className="navbar-breadcrumb">
          LightMamba-ASL
          <ChevronRight />
          <span>{pageLabel}</span>
        </div>
      </div>

      <div className="navbar-right">
        <div className={`navbar-badge badge ${online === true ? 'badge-online' : online === false ? 'badge-offline' : 'badge-warning'}`}>
          <span className={`pulse-dot ${online === true ? 'green' : online === false ? 'red' : 'amber'}`} />
          {online === true ? 'API Online' : online === false ? 'API Offline' : 'Connecting'}
        </div>
      </div>
    </header>
  );
}
