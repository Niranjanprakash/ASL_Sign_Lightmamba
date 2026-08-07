import { useState, useEffect, useCallback } from 'react';
import { getHealth } from '../services/api';
import { HEALTH_CHECK_INTERVAL_MS } from '../utils/constants';

export function useBackendStatus() {
  const [online, setOnline]     = useState(null); // null = unknown
  const [checking, setChecking] = useState(false);

  const check = useCallback(async () => {
    setChecking(true);
    try {
      await getHealth();
      setOnline(true);
    } catch {
      setOnline(false);
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    check();
    const id = setInterval(check, HEALTH_CHECK_INTERVAL_MS);
    return () => clearInterval(id);
  }, [check]);

  return { online, checking, refresh: check };
}
