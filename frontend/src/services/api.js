/**
 * API service for communicating with backend services.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const SPRINGBOOT_URL = import.meta.env.VITE_SPRINGBOOT_URL || 'http://localhost:8080';

/**
 * FastAPI backend service
 */
export const fastApiService = {
  async infer(metrics) {
    const response = await fetch(`${API_BASE_URL}/api/v1/infer`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(metrics),
    });

    if (!response.ok) {
      throw new Error(`Inference failed: ${response.statusText}`);
    }

    return response.json();
  },

  async getConfig() {
    const response = await fetch(`${API_BASE_URL}/config`);

    if (!response.ok) {
      throw new Error(`Failed to get config: ${response.statusText}`);
    }

    return response.json();
  },

  async updateConfig(config) {
    const response = await fetch(`${API_BASE_URL}/config`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      throw new Error(`Failed to update config: ${response.statusText}`);
    }

    return response.json();
  },

  async health() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  },
};

/**
 * Spring Boot service
 */
export const springBootService = {
  async register(userData) {
    const response = await fetch(`${SPRINGBOOT_URL}/api/v1/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(userData),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: response.statusText }));
      throw new Error(error.message || 'Registration failed');
    }

    return response.json();
  },

  async login(credentials) {
    const response = await fetch(`${SPRINGBOOT_URL}/api/v1/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(credentials),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: response.statusText }));
      throw new Error(error.message || 'Login failed');
    }

    return response.json();
  },

  async infer(metrics, token) {
    const response = await fetch(`${SPRINGBOOT_URL}/api/v1/infer`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify(metrics),
    });

    if (!response.ok) {
      throw new Error(`Inference failed: ${response.statusText}`);
    }

    return response.json();
  },

  async activateSOS(sosData, token) {
    const response = await fetch(`${SPRINGBOOT_URL}/api/v1/sos/activate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify(sosData),
    });

    if (!response.ok) {
      throw new Error(`SOS activation failed: ${response.statusText}`);
    }

    return response.json();
  },

  async getFamilyMembers(token) {
    const response = await fetch(`${SPRINGBOOT_URL}/api/v1/family/members`, {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to get family members: ${response.statusText}`);
    }

    return response.json();
  },

  async getDriverStatus(driverId, token) {
    const response = await fetch(`${SPRINGBOOT_URL}/api/v1/family/driver-status/${driverId}`, {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to get driver status: ${response.statusText}`);
    }

    return response.json();
  },
};

/**
 * WebSocket service for real-time notifications
 */
export class WebSocketService {
  constructor(url) {
    this.url = url || `${SPRINGBOOT_URL.replace('http', 'ws')}/ws`;
    this.stompClient = null;
    this.connected = false;
    this.subscriptions = new Map();
  }

  connect(onConnect, onError) {
    if (typeof SockJS === 'undefined' || typeof Stomp === 'undefined') {
      console.warn('SockJS and Stomp.js required for WebSocket support');
      return;
    }

    const socket = new SockJS(this.url);
    this.stompClient = Stomp.over(socket);

    this.stompClient.connect(
      {},
      () => {
        this.connected = true;
        if (onConnect) onConnect();
      },
      (error) => {
        this.connected = false;
        if (onError) onError(error);
      }
    );
  }

  subscribe(topic, callback) {
    if (!this.connected || !this.stompClient) {
      console.warn('WebSocket not connected');
      return null;
    }

    const subscription = this.stompClient.subscribe(topic, (message) => {
      const data = JSON.parse(message.body);
      callback(data);
    });

    this.subscriptions.set(topic, subscription);
    return subscription;
  }

  unsubscribe(topic) {
    const subscription = this.subscriptions.get(topic);
    if (subscription) {
      subscription.unsubscribe();
      this.subscriptions.delete(topic);
    }
  }

  disconnect() {
    if (this.stompClient && this.connected) {
      this.stompClient.disconnect();
      this.connected = false;
      this.subscriptions.clear();
    }
  }
}


