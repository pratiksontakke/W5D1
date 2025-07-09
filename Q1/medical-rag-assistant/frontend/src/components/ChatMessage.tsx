
import { Message } from '../types';
import SourcePill from './SourcePill';

interface ChatMessageProps {
  message: Message;
}

const ChatMessage = ({ message }: ChatMessageProps) => {
  if (message.role === 'loading') {
    return (
      <div className="flex justify-start mb-4">
        <div className="max-w-xs lg:max-w-md px-4 py-3 bg-gray-100 rounded-2xl">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
          </div>
        </div>
      </div>
    );
  }

  if (message.role === 'user') {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-xs lg:max-w-md px-4 py-3 bg-blue-600 text-white rounded-2xl">
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-xs lg:max-w-md">
        <div className="px-4 py-3 bg-gray-100 rounded-2xl">
          <p className="text-sm text-gray-900 whitespace-pre-wrap">{message.content}</p>
        </div>
        {message.sources && message.sources.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {message.sources.map((source) => (
              <SourcePill key={source.id} source={source} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
