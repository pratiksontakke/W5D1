
import { Paperclip } from 'lucide-react';

const DocumentUploader = () => {
  const handleUploadClick = () => {
    console.log("Upload clicked");
  };

  return (
    <button
      onClick={handleUploadClick}
      className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-full transition-colors duration-200"
      title="Upload document"
    >
      <Paperclip className="w-5 h-5" />
    </button>
  );
};

export default DocumentUploader;
