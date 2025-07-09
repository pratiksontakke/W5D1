
import { Link, useLocation } from 'react-router-dom';
import { MessageSquare, Upload } from 'lucide-react';

const Header = () => {
  const location = useLocation();

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4 fixed top-0 left-0 right-0 z-10">
      <div className="max-w-4xl mx-auto flex items-center justify-between">
        <h1 className="text-xl font-semibold text-gray-900">
          Medical Knowledge Assistant
        </h1>
        
        <nav className="flex space-x-4">
          <Link
            to="/"
            className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${
              location.pathname === '/'
                ? 'bg-blue-100 text-blue-700'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
            }`}
          >
            <MessageSquare className="w-4 h-4" />
            <span>Chat</span>
          </Link>
          <Link
            to="/upload"
            className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${
              location.pathname === '/upload'
                ? 'bg-blue-100 text-blue-700'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
            }`}
          >
            <Upload className="w-4 h-4" />
            <span>Upload & Train</span>
          </Link>
        </nav>
      </div>
    </header>
  );
};

export default Header;
