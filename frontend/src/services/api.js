const API_BASE_URL = 'http://localhost:8000'

export const chatAPI = {
  sendMessage: async (message) => {
    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      return data
    } catch (error) {
      console.error('API call failed:', error)
      throw error
    }
  },

  getModels: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/models`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      return await response.json()
    } catch (error) {
      console.error('Failed to fetch models:', error)
      throw error
    }
  }
}