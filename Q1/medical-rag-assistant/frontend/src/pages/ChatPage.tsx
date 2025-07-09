
import { useState } from 'react';
import { Message } from '../types';
import Header from '../components/Header';
import MessageList from '../components/MessageList';
import ChatInput from '../components/ChatInput';

const ChatPage = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async (content: string) => {
    // Add user message immediately
    const userMessage: Message = {
      role: 'user',
      content,
    };

    // Add loading message
    const loadingMessage: Message = {
      role: 'loading',
      content: '',
    };

    setMessages(prev => [...prev, userMessage, loadingMessage]);
    setIsLoading(true);

    try {
      // Call API endpoint
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: content }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();

      // Create AI response message
      const aiResponse: Message = {
        role: 'assistant',
        content: data.response,
        sources: data.sources || [],
      };

      // Replace loading message with AI response
      setMessages(prev => [...prev.slice(0, -1), aiResponse]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Fallback response for demo purposes
      const errorResponse: Message = {
        role: 'assistant',
        content: `Thank you for your question: "${content}". This is a simulated response from the Medical Knowledge Assistant. In a real implementation, this would connect to a medical AI service to provide accurate, evidence-based medical information.`,
        sources: [
          { id: '1', name: 'Medical Journal' },
          { id: '2', name: 'Clinical Guidelines' },
        ],
      };

      setMessages(prev => [...prev.slice(0, -1), errorResponse]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <Header />
      <div className="flex-1 pt-20 pb-24 overflow-hidden">
        <MessageList messages={messages} />
      </div>
      <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
    </div>
  );
};

export default ChatPage;
