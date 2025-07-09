
export interface Source {
  id: string;
  name: string;
}

export interface Message {
  role: 'user' | 'assistant' | 'loading';
  content: string;
  sources?: Source[];
}
