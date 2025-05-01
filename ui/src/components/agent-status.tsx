import { useState, useEffect } from 'react';

interface AgentStatusProps {
  className?: string;
}

export default function AgentStatus({ className = '' }: AgentStatusProps) {
  const [status, setStatus] = useState<'connected' | 'disconnected' | 'loading'>('loading');
  const [statusText, setStatusText] = useState<string>('Checking connection...');

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await fetch('/api/agent/status');
        if (response.ok) {
          const data = await response.json();
          if (data.status === 'running') {
            setStatus('connected');
            const modelInfo = data.provider && data.model ? 
              `${data.provider}/${data.model}` : 
              'Unknown model';
            setStatusText(`Connected: ${modelInfo}`);
          } else {
            setStatus('disconnected');
            setStatusText('Agent unavailable');
          }
        } else {
          setStatus('disconnected');
          setStatusText('Connection error');
        }
      } catch (error) {
        setStatus('disconnected');
        setStatusText('Connection error');
      }
    };

    // Check status immediately
    checkStatus();
    
    // Set up interval to check status periodically
    const interval = setInterval(checkStatus, 30000); // Check every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  // Determine color and icon based on status
  const getStatusColor = () => {
    switch (status) {
      case 'connected': return 'text-green-500';
      case 'disconnected': return 'text-red-500';
      case 'loading': default: return 'text-yellow-500';
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'connected': return '●';
      case 'disconnected': return '●';
      case 'loading': default: return '○';
    }
  };

  return (
    <div className={`flex items-center ${className}`}>
      <span className={`mr-1.5 text-sm ${getStatusColor()}`}>
        {getStatusIcon()}
      </span>
      <span className="text-xs opacity-70">{statusText}</span>
    </div>
  );
}