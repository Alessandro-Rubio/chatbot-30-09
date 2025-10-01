const API_BASE_URL = 'http://localhost:8000';

export const chatAPI = {
  sendMessage: async (message, useRAG = true) => {
    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, use_rag: useRAG })
      });
      
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }
};

export const fileAPI = {
  uploadFile: async (formData) => {
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  },

  deleteFile: async (filename) => {
    const response = await fetch(`${API_BASE_URL}/files/${filename}`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  },

  listFiles: async () => {
    const response = await fetch(`${API_BASE_URL}/files`);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    return await response.json();
  }
};

export const ragAPI = {
  getStatus: async () => {
    const response = await fetch(`${API_BASE_URL}/rag/status`);
    return await response.json();
  },

  reinitialize: async () => {
    const response = await fetch(`${API_BASE_URL}/rag/reinitialize`, {
      method: 'POST'
    });
    return await response.json();
  }
};