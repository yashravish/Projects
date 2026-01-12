const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

export async function apiRequest<T>(
  endpoint: string,
  options?: RequestInit
): Promise<{ data?: T; error?: { code: string; message: string; details?: unknown } }> {
  const url = `${API_URL}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  const result = await response.json();

  if (!response.ok) {
    return { error: result.error };
  }

  return result;
}

export async function uploadFile(
  endpoint: string,
  file: File
): Promise<{ data?: unknown; error?: { code: string; message: string } }> {
  const url = `${API_URL}${endpoint}`;
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(url, {
    method: 'POST',
    credentials: 'include',
    body: formData,
  });

  const result = await response.json();

  if (!response.ok) {
    return { error: result.error };
  }

  return result;
}
