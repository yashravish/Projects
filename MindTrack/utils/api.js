// src/utils/api.js

export const fetchData = async (endpoint) => {
    try {
      const response = await fetch(endpoint);
      return await response.json();
    } catch (error) {
      console.error("API fetch error:", error);
      return null;
    }
  };
  