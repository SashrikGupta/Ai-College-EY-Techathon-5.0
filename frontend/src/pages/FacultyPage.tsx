import React, { useContext, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle } from 'lucide-react';
import { CurrConfigContext } from '../context.tsx';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

type GeneratedTest = {
  format: {
    context_title: string;
    question: string;
  }[];
};

const FacultyPage = () => {
  const cont = useContext(CurrConfigContext) || {};

  // State for file upload, prompt text, processing, errors, and the generated test result
  const [files, setFiles] = React.useState<File[]>([]);
  const [prompt, setPrompt] = React.useState('');
  const [processing, setProcessing] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  // For demonstration, pre-populated test for UI testing
  // In production, start with null and set this after the API call to generate the test.
  const [generatedTest, setGeneratedTest] = React.useState<GeneratedTest | null>();

  // State to hold user answers for each question.
  const [answers, setAnswers] = React.useState<string[]>([]);
  useEffect(() => {
    if (generatedTest) {
      setAnswers(generatedTest.format.map(() => ''));
    }
  }, [generatedTest]);

  // Callback for file upload via dropzone
  const onDrop = React.useCallback((acceptedFiles: File[]) => {
    setFiles(acceptedFiles);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    maxFiles: 1,
  });

  // Call the API to generate the test based on the uploaded PDF and prompt.
  const handleProcess = async () => {
    if (files.length === 0 || prompt.trim() === '') {
      setError('Please upload a PDF and provide a context.');
      return;
    }

    setProcessing(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('pdf', files[0]);
      formData.append('data', JSON.stringify({ prompt }));

      const requestOptions = {
        method: 'POST',
        body: formData,
        redirect: 'follow' as RequestRedirect,
      };

      const response = await fetch(
        'http://127.0.0.1:5000/teacher/generate_test',
        requestOptions
      );
      if (!response.ok) {
        console.log(response)
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result: GeneratedTest = await response.json();
      setGeneratedTest(result);
    } catch (err) {
      console.error(err);
      setError('An error occurred while generating the test. Please try again.');
    } finally {
      setProcessing(false);
    }
  };

  // Reset UI back to the upload state
  const handleReset = () => {
    setFiles([]);
    setPrompt('');
    setGeneratedTest(null);
    setError(null);
  };

  // Submit answers: construct answerSheet JSON, call the evaluate_test API,
  // show a toast on success or error, and on success reset to the upload state.
  const handleSubmitAnswers = async () => {
    if (generatedTest) {
      const answerSheet = generatedTest.format.map((item, index) => ({
        context_title: item.context_title,
        question: item.question,
        answer: answers[index],
      }));

      setProcessing(true);
      try {
        const myHeaders = new Headers();
        myHeaders.append('Content-Type', 'application/json');

        const payload = {
          userid: cont?.user._id,
          answer_sheet: answerSheet,
        };

        const requestOptions = {
          method: 'POST',
          headers: myHeaders,
          body: JSON.stringify(payload),
          redirect: 'follow' as RequestRedirect,
        };

        const response = await fetch(
          'http://127.0.0.1:5000/teacher/evaluate_test',
          requestOptions
        );
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.text();
        console.log(result);
        toast.success('Report generated, see your results in the reports section');
        // Move back to upload state after successful evaluation
        handleReset();
      } catch (error) {
        console.error(error);
        toast.error('Error occurred while evaluating test. Please click again');
      } finally {
        setProcessing(false);
      }
    }
  };

  // --- State 2: Display the generated test with writing areas for answers ---
  if (generatedTest) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <ToastContainer />
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Generated Test</h2>
          <div className="space-y-6">
            {generatedTest.format.map((item, index) => (
              <div key={index} className="p-4 border rounded-lg bg-gray-50">
                <h3 className="font-semibold text-gray-800">{item.context_title}</h3>
                <p className="text-gray-600 mt-1">{item.question}</p>
                <textarea
                  value={answers[index] || ''}
                  onChange={(e) => {
                    const newAnswers = [...answers];
                    newAnswers[index] = e.target.value;
                    setAnswers(newAnswers);
                  }}
                  placeholder="Write your answer here..."
                  className="mt-2 p-2 border rounded w-full"
                  rows={3}
                ></textarea>
              </div>
            ))}
          </div>
          <div className="mt-6 flex space-x-4">
            <button
              onClick={handleSubmitAnswers}
              disabled={processing}
              className="px-4 py-2 rounded-md bg-green-500 hover:bg-green-600 text-white"
            >
              {processing ? 'Uploading Answer...' : 'Submit Answers'}
            </button>
            <button
              onClick={handleReset}
              disabled={processing}
              className="px-4 py-2 rounded-md bg-primary-500 hover:bg-primary-600 text-white"
            >
              Upload New Test
            </button>
          </div>
        </div>
      </div>
    );
  }


  if(processing){
    return(

<div className="h-[90vh] w-[100vw] flex flex-col items-center justify-center">
  <div className="typewriter">
    <div className="slide"><i></i></div>
    <div className="paper"></div>
    <div className="keyboard"></div>
  </div>

  <div className="carder">
    <div className="loader_carder text-center">
    <div className="words">
      <span className="word"> searching in the pdf </span>
      <span className="word"> finding info from web</span>
      <span className="word"> processing the info </span>
      <span className="word"> gatthering data </span>
      <span className="word"> processing all the context </span>
    </div>
    </div>
  </div>
</div>
  
    )
  }

  // --- State 1: Upload PDF and provide test context ---
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <ToastContainer />
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Generate Test from Course Material</h2>
        <p className="text-gray-600 mb-8">
          Upload your PDF and provide context for the test generation.
        </p>

        {/* File Upload */}
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-500'
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="mx-auto h-12 w-12 text-gray-400" />
          <p className="mt-2 text-sm text-gray-600">
            Drag and drop your PDF here, or click to select
          </p>
          <p className="text-xs text-gray-500 mt-1">Maximum file size: 10MB</p>
        </div>

        {/* Display Selected File */}
        {files.length > 0 && (
          <div className="mt-6">
            <div className="flex items-center p-4 bg-gray-50 rounded-lg">
              <FileText className="h-6 w-6 text-gray-400" />
              <span className="ml-2 text-sm text-gray-900">{files[0].name}</span>
            </div>
          </div>
        )}

        {/* Context/Prompt Input */}
        <div className="mt-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Context for Test Generation
          </label>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g., full stack development"
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500"
          />
        </div>

        {/* Error Message */}
        {error && (
          <div className="mt-4 p-4 rounded-lg bg-red-50">
            <div className="flex items-center">
              <AlertCircle className="h-5 w-5 text-red-500" />
              <span className="ml-2 text-sm text-red-700">{error}</span>
            </div>
          </div>
        )}

        {/* Generate Test Button */}
        <button
          onClick={handleProcess}
          disabled={processing}
          className={`mt-4 w-full px-4 py-2 rounded-md text-white ${
            processing ? 'bg-gray-400 cursor-not-allowed' : 'bg-primary-500 hover:bg-primary-600'
          }`}
        >
          {processing ? 'Generating Test...' : 'Generate Test'}
        </button>
      </div>
    </div>
  );
};

export default FacultyPage;
