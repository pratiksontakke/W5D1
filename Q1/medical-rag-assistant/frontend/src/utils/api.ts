
// API endpoints configuration
export const API_ENDPOINTS = {
  CHAT: '/api/chat',
  UPLOAD_DOCUMENT: '/api/upload-document',
} as const;

// API response types
export interface ChatResponse {
  response: string;
  sources?: Array<{
    id: string;
    name: string;
  }>;
}

export interface UploadResponse {
  success: boolean;
  message: string;
  documentId?: string;
}

// API error handling
export class APIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public code?: string
  ) {
    super(message);
    this.name = 'APIError';
  }
}

// Helper function for API calls
export async function apiCall<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  try {
    const response = await fetch(endpoint, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new APIError(
        `API call failed: ${response.statusText}`,
        response.status
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError('Network error occurred');
  }
}
