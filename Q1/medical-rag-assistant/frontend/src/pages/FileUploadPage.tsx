
import { useState } from 'react';
import { Upload, File, CheckCircle, AlertCircle, X } from 'lucide-react';
import Header from '../components/Header';

interface UploadedFile {
  file: File;
  status: 'pending' | 'processing' | 'completed' | 'error';
  id: string;
}

const FileUploadPage = () => {
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFiles(Array.from(e.target.files));
    }
  };

  const handleFiles = (files: File[]) => {
    const newFiles: UploadedFile[] = files.map(file => ({
      file,
      status: 'pending',
      id: Math.random().toString(36).substr(2, 9),
    }));
    
    setUploadedFiles(prev => [...prev, ...newFiles]);
  };

  const removeFile = (id: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== id));
  };

  const processFiles = async () => {
    setIsProcessing(true);
    
    // Update all pending files to processing
    setUploadedFiles(prev => 
      prev.map(f => f.status === 'pending' ? { ...f, status: 'processing' } : f)
    );

    try {
      for (const uploadedFile of uploadedFiles.filter(f => f.status === 'processing')) {
        const formData = new FormData();
        formData.append('file', uploadedFile.file);

        try {
          const response = await fetch('/api/upload-document', {
            method: 'POST',
            body: formData,
          });

          if (!response.ok) {
            throw new Error('Upload failed');
          }

          const data = await response.json();
          console.log('File processed successfully:', data);

          // Update file status to completed
          setUploadedFiles(prev => 
            prev.map(f => 
              f.id === uploadedFile.id 
                ? { ...f, status: 'completed' } 
                : f
            )
          );
        } catch (error) {
          console.error('Error processing file:', uploadedFile.file.name, error);
          
          // Update file status to error
          setUploadedFiles(prev => 
            prev.map(f => 
              f.id === uploadedFile.id 
                ? { ...f, status: 'error' } 
                : f
            )
          );
        }

        // Small delay between files
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const getStatusIcon = (status: UploadedFile['status']) => {
    switch (status) {
      case 'pending':
        return <File className="h-5 w-5 text-gray-400" />;
      case 'processing':
        return <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>;
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-600" />;
      default:
        return <File className="h-5 w-5 text-gray-400" />;
    }
  };

  const getStatusText = (status: UploadedFile['status']) => {
    switch (status) {
      case 'pending':
        return 'Ready to process';
      case 'processing':
        return 'Generating embeddings...';
      case 'completed':
        return 'Training completed';
      case 'error':
        return 'Processing failed';
      default:
        return '';
    }
  };

  const pendingFiles = uploadedFiles.filter(f => f.status === 'pending');
  const hasCompletedFiles = uploadedFiles.some(f => f.status === 'completed');

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <Header />
      <div className="flex-1 pt-20 p-6">
        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-lg shadow-sm p-8">
            <h2 className="text-2xl font-semibold text-gray-900 mb-6">
              Upload Documents for Training
            </h2>
            <p className="text-gray-600 mb-8">
              Upload medical documents, research papers, or other files to train the AI assistant. 
              The system will generate embeddings from your documents to improve response accuracy.
            </p>

            {/* File Upload Area */}
            <div
              className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                dragActive
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-300 hover:border-gray-400'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                type="file"
                multiple
                onChange={handleChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                accept=".pdf,.doc,.docx,.txt,.md"
                disabled={isProcessing}
              />
              <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <p className="text-lg font-medium text-gray-900 mb-2">
                Drop files here or click to upload
              </p>
              <p className="text-sm text-gray-500">
                Supports PDF, DOC, DOCX, TXT, MD files
              </p>
            </div>

            {/* Uploaded Files List */}
            {uploadedFiles.length > 0 && (
              <div className="mt-8">
                <h3 className="text-lg font-medium text-gray-900 mb-4">
                  Files ({uploadedFiles.length})
                </h3>
                <div className="space-y-3">
                  {uploadedFiles.map((uploadedFile) => (
                    <div
                      key={uploadedFile.id}
                      className="flex items-center space-x-3 p-4 bg-gray-50 rounded-lg"
                    >
                      {getStatusIcon(uploadedFile.status)}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {uploadedFile.file.name}
                        </p>
                        <p className="text-xs text-gray-500">
                          {(uploadedFile.file.size / 1024 / 1024).toFixed(2)} MB • {getStatusText(uploadedFile.status)}
                        </p>
                      </div>
                      {uploadedFile.status === 'pending' && (
                        <button
                          onClick={() => removeFile(uploadedFile.id)}
                          className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Process Files Button */}
            {pendingFiles.length > 0 && (
              <div className="mt-8">
                <button
                  onClick={processFiles}
                  disabled={isProcessing}
                  className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors duration-200 flex items-center justify-center space-x-2"
                >
                  {isProcessing ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                      <span>Processing files and generating embeddings...</span>
                    </>
                  ) : (
                    <>
                      <CheckCircle className="h-5 w-5" />
                      <span>Process {pendingFiles.length} File{pendingFiles.length > 1 ? 's' : ''} & Generate Embeddings</span>
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Success Message */}
            {hasCompletedFiles && !isProcessing && pendingFiles.length === 0 && (
              <div className="mt-8 p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-5 w-5 text-green-600" />
                  <p className="text-green-800 font-medium">
                    Training completed successfully! You can now upload more documents or start chatting.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default FileUploadPage;
