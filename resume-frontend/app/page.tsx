'use client'
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function ResumeAnalyzer() {
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  // Initialize client-side state after hydration
  useEffect(() => {
    // This will run only on the client side after hydration
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setIsLoading(true);
    setAnalysis('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/analyze-resume', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok || !response.body) {
        throw new Error('Failed to analyze resume');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        setAnalysis((prev) => prev + chunk);
      }

    } catch (err) {
      console.error(err);
      setAnalysis('Something went wrong while analyzing the resume.');
    } finally {
      setIsLoading(false);
    }
  };

  // Function to format the analysis text with proper styling
  const formatAnalysis = (text: string) => {
    if (!text) return null;
    
    // Split by numbered sections
    const sections = text.split(/\n\s*\d+\./).filter(section => section.trim() !== '');
    
    return sections.map((section, index) => {
      const [title, ...content] = section.trim().split('\n');
      const contentText = content.join('\n').trim();
      
      return (
        <div key={index} className="mb-6">
          <h3 className="text-xl font-semibold text-indigo-700 mb-2">{title}</h3>
          {contentText.split('\n').map((line, i) => (
            <p key={i} className="text-gray-700 mb-2">
              {line.startsWith('-') ? '• ' + line.substring(1) : line}
            </p>
          ))}
        </div>
      );
    });
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <main className="flex-grow container mx-auto px-4 py-8 max-w-4xl">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-indigo-800 mb-2">
            Resume Analyzer
          </h1>
          <p className="text-gray-600">
            Upload your resume to get detailed analysis and improvement suggestions
          </p>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6 mb-8 border border-gray-100">
          <div className="space-y-4">
            <div>
              <label htmlFor="resume" className="block text-sm font-medium text-gray-700">
                Upload Resume (PDF)
              </label>
              <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                <div className="space-y-1 text-center">
                  <svg
                    className="mx-auto h-12 w-12 text-gray-400"
                    stroke="currentColor"
                    fill="none"
                    viewBox="0 0 48 48"
                    aria-hidden="true"
                  >
                    <path
                      d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                      strokeWidth={2}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <div className="flex text-sm text-gray-600">
                    <label
                      htmlFor="resume"
                      className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500"
                    >
                      <span>Upload a file</span>
                      <input
                        id="resume"
                        name="resume"
                        type="file"
                        accept=".pdf"
                        onChange={handleFileChange}
                        className="sr-only"
                      />
                    </label>
                    <p className="pl-1">or drag and drop</p>
                  </div>
                  <p className="text-xs text-gray-500">
                    PDF files only
                  </p>

                  {file && (
                    <p className="text-sm text-gray-700 mt-2">
                      <strong>Selected File:</strong> {file.name}
                    </p>
                  )}
                </div>
              </div>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={!file || isLoading}
              className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 ${
                !file || isLoading ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {isLoading ? 'Analyzing...' : 'Analyze Resume'}
            </button>
          </div>
        </div>

        {isLoading && (
          <div className="mt-6 p-6 text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-indigo-500 mb-4"></div>
            <p className="text-gray-600">Analyzing your resume...</p>
          </div>
        )}
          
        {analysis && !isLoading && (
          <div className="mt-8">
            <div className="border-b border-gray-200 pb-4 mb-6">
              <h2 className="text-2xl font-bold text-gray-900">Analysis Results</h2>
              <p className="text-sm text-gray-500 mt-1">
                Based on your resume, here's our detailed analysis
              </p>
            </div>
            <div className="space-y-6">
              {formatAnalysis(analysis)}
            </div>
            <div className="mt-8 pt-6 border-t border-gray-200">
              <p className="text-sm text-gray-500 text-center">
                Remember to save these suggestions for your next resume update!
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}