import { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { updateAgentConfig } from '@/lib/agent-integration';

// Define provider options with models
const PROVIDER_OPTIONS = [
  {
    id: 'bedrock',
    name: 'AWS Bedrock',
    models: [
      { id: 'anthropic.claude-3-7-sonnet-20250219-v1:0', name: 'Claude 3.7 Sonnet' },
      { id: 'anthropic.claude-3-5-sonnet-20240620-v1:0', name: 'Claude 3.5 Sonnet' },
      { id: 'amazon.titan-text-express-v1', name: 'Amazon Titan Express' },
      { id: 'meta.llama3-8b-instruct-v1:0', name: 'Meta Llama 3 8B' },
      { id: 'meta.llama3-70b-instruct-v1:0', name: 'Meta Llama 3 70B' },
    ]
  },
  {
    id: 'openai',
    name: 'OpenAI',
    models: [
      { id: 'gpt-4o', name: 'GPT-4o' },
      { id: 'gpt-4-turbo', name: 'GPT-4 Turbo' },
      { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo' },
    ]
  },
  {
    id: 'anthropic',
    name: 'Anthropic',
    models: [
      { id: 'claude-3-7-sonnet-latest', name: 'Claude 3.7 Sonnet' },
      { id: 'claude-3-opus-latest', name: 'Claude 3 Opus' },
      { id: 'claude-3-sonnet-latest', name: 'Claude 3 Sonnet' },
      { id: 'claude-3-haiku-latest', name: 'Claude 3 Haiku' },
    ]
  },
  {
    id: 'google',
    name: 'Google',
    models: [
      { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro' },
      { id: 'gemini-1.5-flash', name: 'Gemini 1.5 Flash' },
      { id: 'gemini-1.0-pro', name: 'Gemini 1.0 Pro' },
    ]
  },
  {
    id: 'groq',
    name: 'Groq',
    models: [
      { id: 'llama3-70b-8192', name: 'Llama 3 70B' },
      { id: 'llama3-8b-8192', name: 'Llama 3 8B' },
      { id: 'mixtral-8x7b-32768', name: 'Mixtral 8x7B' },
    ]
  },
  {
    id: 'azure',
    name: 'Azure OpenAI',
    models: [
      { id: 'gpt-4', name: 'GPT-4' },
      { id: 'gpt-35-turbo', name: 'GPT-3.5 Turbo' },
    ]
  },
  {
    id: 'huggingface',
    name: 'HuggingFace',
    models: [
      { id: 'mistralai/Mixtral-8x7B-Instruct-v0.1', name: 'Mixtral 8x7B' },
      { id: 'meta-llama/Llama-2-70b-chat-hf', name: 'Llama 2 70B' },
    ]
  },
];

// Temperature options
const TEMPERATURE_OPTIONS = [
  { value: 0, label: '0 - Deterministic' },
  { value: 0.2, label: '0.2 - Conservative' },
  { value: 0.5, label: '0.5 - Balanced' },
  { value: 0.7, label: '0.7 - Creative' },
  { value: 1, label: '1.0 - Random' },
];

interface ProviderSelectorProps {
  onProviderChange?: (provider: string, model: string) => void;
  className?: string;
}

export default function ProviderSelector({ 
  onProviderChange, 
  className = '' 
}: ProviderSelectorProps) {
  // State for current selections
  const [selectedProvider, setSelectedProvider] = useState('bedrock');
  const [selectedModel, setSelectedModel] = useState('anthropic.claude-3-7-sonnet-20250219-v1:0');
  const [temperature, setTemperature] = useState(0.2);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [currentProvider, setCurrentProvider] = useState('');
  const [currentModel, setCurrentModel] = useState('');

  // Current provider's models
  const currentProviderData = PROVIDER_OPTIONS.find(p => p.id === selectedProvider);
  const availableModels = currentProviderData?.models || [];

  // Fetch current provider info on mount
  useEffect(() => {
    const fetchProviderInfo = async () => {
      try {
        const response = await fetch('/api/agent/status');
        if (response.ok) {
          const data = await response.json();
          if (data.provider && data.model) {
            setCurrentProvider(data.provider);
            setCurrentModel(data.model);
            // Update selected values to match current
            setSelectedProvider(data.provider);
            setSelectedModel(data.model);
          }
        }
      } catch (error) {
        console.error('Failed to fetch provider info:', error);
      }
    };

    fetchProviderInfo();
  }, []);

  // Handle provider change
  const handleProviderChange = (providerId: string) => {
    setSelectedProvider(providerId);
    // Set default model for this provider
    const provider = PROVIDER_OPTIONS.find(p => p.id === providerId);
    if (provider && provider.models.length > 0) {
      setSelectedModel(provider.models[0].id);
    }
  };

  // Apply changes
  const applyChanges = async () => {
    setIsLoading(true);
    try {
      // Call API to update provider and model
      const result = await updateAgentConfig({
        provider: selectedProvider,
        model: selectedModel,
        temperature: temperature
      });

      if (result.status === 'success') {
        // Update current values
        setCurrentProvider(selectedProvider);
        setCurrentModel(selectedModel);
        
        // Notify parent component
        if (onProviderChange) {
          onProviderChange(selectedProvider, selectedModel);
        }
        
        // Close dropdown
        setIsOpen(false);
      } else {
        console.error('Failed to update provider:', result.message);
        alert('Failed to update provider: ' + result.message);
      }
    } catch (error) {
      console.error('Error updating provider:', error);
      alert('Error updating provider. See console for details.');
    } finally {
      setIsLoading(false);
    }
  };

  // Find provider and model display names
  const getProviderName = (id: string) => {
    return PROVIDER_OPTIONS.find(p => p.id === id)?.name || id;
  };

  const getModelName = (providerId: string, modelId: string) => {
    const provider = PROVIDER_OPTIONS.find(p => p.id === providerId);
    if (!provider) return modelId;
    
    const model = provider.models.find(m => m.id === modelId);
    return model ? model.name : modelId;
  };

  return (
    <div className={`relative ${className}`}>
      {/* Current provider display */}
      <div 
        className="flex items-center justify-between p-2 rounded bg-gray-100 dark:bg-gray-800 cursor-pointer"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center space-x-2">
          <span className="font-medium">🤖</span>
          <span>
            {currentProvider ? 
              `${getProviderName(currentProvider)} / ${getModelName(currentProvider, currentModel)}` : 
              'Select LLM Provider'}
          </span>
        </div>
        <svg 
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} 
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20" 
          fill="currentColor"
        >
          <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
        </svg>
      </div>

      {/* Dropdown menu */}
      {isOpen && (
        <div className="absolute top-full left-0 right-0 mt-1 p-3 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg z-50">
          <div className="mb-4">
            <h3 className="mb-2 font-medium">Provider</h3>
            <div className="grid grid-cols-2 gap-2">
              {PROVIDER_OPTIONS.map(provider => (
                <button
                  key={provider.id}
                  className={`
                    py-1 px-3 text-sm rounded text-left
                    ${selectedProvider === provider.id 
                      ? 'bg-blue-100 dark:bg-blue-800 dark:text-white' 
                      : 'bg-gray-100 dark:bg-gray-800'}
                  `}
                  onClick={() => handleProviderChange(provider.id)}
                >
                  {provider.name}
                </button>
              ))}
            </div>
          </div>

          <div className="mb-4">
            <h3 className="mb-2 font-medium">Model</h3>
            <div className="grid grid-cols-1 gap-2">
              {availableModels.map(model => (
                <button
                  key={model.id}
                  className={`
                    py-1 px-3 text-sm rounded text-left
                    ${selectedModel === model.id 
                      ? 'bg-blue-100 dark:bg-blue-800 dark:text-white' 
                      : 'bg-gray-100 dark:bg-gray-800'}
                  `}
                  onClick={() => setSelectedModel(model.id)}
                >
                  {model.name}
                </button>
              ))}
            </div>
          </div>

          <div className="mb-4">
            <h3 className="mb-2 font-medium">Temperature</h3>
            <select 
              className="w-full p-2 border border-gray-300 dark:border-gray-700 rounded bg-white dark:bg-gray-800"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
            >
              {TEMPERATURE_OPTIONS.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex justify-end space-x-2">
            <Button 
              variant="outline" 
              onClick={() => setIsOpen(false)}
            >
              Cancel
            </Button>
            <Button 
              onClick={applyChanges}
              disabled={isLoading}
            >
              {isLoading ? 'Applying...' : 'Apply'}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}