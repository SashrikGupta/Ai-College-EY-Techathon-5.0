import React from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";
import {
  Upload,
  FileText,
  Presentation,
  Mic,
  Video,
  Image,
  CheckCircle,
  X,
} from "lucide-react";

const ResearchPage = () => {
  // File upload & progress states
  const [files, setFiles] = React.useState<File[]>([]);
  const [uploadProgress, setUploadProgress] = React.useState(0);

  // Which output format is selected ("presentation", "podcast", "video", "comic")
  const [selectedOutput, setSelectedOutput] = React.useState<string | null>(
    null
  );
  // Global processing state (used during transformation API calls)
  const [processing, setProcessing] = React.useState(false);
  

  // ------------------------------
  // Extra state for each transformation

  // Presentation-specific states
  const [researchTitle, setResearchTitle] = React.useState("");
  const [presentationStyle, setPresentationStyle] = React.useState<
    "professional" | "fun"
  >("professional");
  const [pptUrl, setPptUrl] = React.useState<string | null>(null);

  // Podcast-specific states
  const lengthOptions = ["Short (1-2 min)", "Medium (3-5 min)"];
  const toneOptions = ["Fun", "Formal"];
  const languageOptions = [
    "Portuguese",
    "Polish",
    "English",
    "Italian",
    "German",
    "Korean",
    "Russian",
    "Hindi",
    "French",
    "Japanese",
    "Chinese",
  ];
  const [podcastLength, setPodcastLength] = React.useState<string>(
    lengthOptions[0]
  );
  const [podcastTone, setPodcastTone] = React.useState<string>(toneOptions[0]);
  const [podcastLanguage, setPodcastLanguage] = React.useState<string>(
    languageOptions[0]
  );
  const [audioUrl, setAudioUrl] = React.useState<string | null>(null);

  // Shorts-specific states
  const [videoUrl, setVideoUrl] = React.useState<string | null>(null);

  // Comic-specific states
  const [comicUrl, setComicUrl] = React.useState<string | null>(null);

  // ------------------------------
  // Define available output types
  const outputTypes = [
    { id: "presentation", name: "Presentation", icon: Presentation },
    { id: "podcast", name: "Podcast", icon: Mic },
    { id: "video", name: "Video", icon: Video },
    { id: "comic", name: "Comic", icon: Image },
  ];

  // ------------------------------
  // Simulate file upload progress (0 to 100 over ~2 seconds)
  const simulateUpload = () => {
    setUploadProgress(0);
    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + 10;
      });
    }, 200);
  };

  // ------------------------------
  // File drop handling using react-dropzone
  const onDrop = React.useCallback((acceptedFiles: File[]) => {
    setFiles(acceptedFiles); // Assume one file is used per transformation
    simulateUpload();
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
  });

  // ------------------------------
  // Toggle output format selection (only one allowed)
  const toggleOutput = (id: string) => {
    setSelectedOutput((prev) => (prev === id ? null : id));
    // Clear previous results when switching types
    setPptUrl(null);
    setAudioUrl(null);
    setVideoUrl(null);
    setComicUrl(null);
  };

  // Remove a file by name
  const removeFile = (name: string) => {
    setFiles((prev) => prev.filter((file) => file.name !== name));
  };

  // ------------------------------
  // POST request functions for each output type

  // For Presentation
  const handleGeneratePPT = async () => {
    if (!files[0] || !researchTitle) {
      alert("Please upload a PDF and enter a research title.");
      return;
    }
    setProcessing(true);
    const formData = new FormData();
    formData.append("pdf_file", files[0]);
    formData.append("research_title", researchTitle);
    const professionalValue = presentationStyle === "professional" ? "1" : "0";
    formData.append("professional", professionalValue);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8080/generate_ppt",
        formData
      );
      if (response.data.ppt_url) {
        setPptUrl(response.data.ppt_url);
      } else {
        alert("Failed to generate PowerPoint.");
      }
    } catch (error) {
      console.error("Error generating PPT:", error);
      alert("Failed to generate the presentation.");
    } finally {
      setProcessing(false);
    }
  };

  // For Podcast
  const handlePodcastProcess = async () => {
    if (!files[0]) return;
    setProcessing(true);
    const formData = new FormData();
    formData.append("file", files[0]);
    formData.append("length", podcastLength);
    formData.append("tone", podcastTone);
    formData.append("language", podcastLanguage);
    formData.append("use_advanced_audio", "true");

    try {
      const response = await fetch("http://localhost:5000/generate_podcast", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      if (result.status === "completed") {
        setAudioUrl(result.audio_url);
      } else {
        alert("Failed to process podcast.");
      }
    } catch (error) {
      console.error("Error processing podcast:", error);
    }
    setProcessing(false);
  };

  // For Shorts (Video)
  const handleShortsProcess = async () => {
    if (!files[0]) return;
    setProcessing(true);
    const formData = new FormData();
    formData.append("pdf", files[0]);

    try {
      const response = await fetch("http://127.0.0.1:5000/process_pdf", {
        method: "POST",
        body: formData,
      });
      if (response.ok) {
        const videoBlob = await response.blob();
        const url = URL.createObjectURL(videoBlob);
        setVideoUrl(url);
      } else {
        console.error("Error processing the file for shorts");
      }
    } catch (error) {
      console.error("Error:", error);
    }
    setProcessing(false);
  };

  // For Comic
  const handleComicProcess = async () => {
    if (!files[0]) return;
    setProcessing(true);
    const formData = new FormData();
    formData.append("file", files[0]);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      if (result.error) {
        console.error("Error:", result.error);
      } else {
        setComicUrl(result.result);
      }
    } catch (error) {
      console.error("Request failed:", error);
    } finally {
      setProcessing(false);
    }
  };

  // Unified handler that calls the appropriate function based on the selected output
  const handleTransform = async () => {
    if (!selectedOutput) return;
    if (selectedOutput === "presentation") {
      await handleGeneratePPT();
    } else if (selectedOutput === "podcast") {
      await handlePodcastProcess();
    } else if (selectedOutput === "video") {
      await handleShortsProcess();
    } else if (selectedOutput === "comic") {
      await handleComicProcess();
    }
  };

  // ------------------------------
  // Render
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">ResearchGen</h2>
        <p className="text-gray-600 mb-8">
          Transform your research papers into engaging multimedia content
        </p>

        {/* File Upload */}
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            isDragActive
              ? "border-[#382D76] bg-[#382D76]/10"
              : "border-gray-300 hover:border-[#382D76]"
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="mx-auto h-12 w-12 text-gray-400" />
          <p className="mt-2 text-sm text-gray-600">
            Drag and drop your research papers here, or click to select
          </p>
          <p className="text-xs text-gray-500 mt-1">Supports PDF format only</p>
        </div>

        {/* Upload Progress Bar & Success Message */}
        {uploadProgress > 0 && uploadProgress < 100 && (
          <div className="w-full bg-gray-200 rounded-full h-2 mb-4 mt-4">
            <div
              className="bg-[#382D76] h-2 rounded-full"
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
        )}
        {uploadProgress === 100 && (
          <div className="flex items-center gap-2 text-green-600 text-sm mt-2">
            <CheckCircle className="h-4 w-4" />
            File uploaded successfully
          </div>
        )}

        {/* Selected Files */}
        {files.length > 0 && (
          <div className="mt-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Selected Papers
            </h3>
            <div className="space-y-2">
              {files.map((file) => (
                <div
                  key={file.name}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg shadow-sm"
                >
                  <div className="flex items-center">
                    <FileText className="h-5 w-5 text-gray-400 mr-2" />
                    <span className="text-sm text-gray-900">{file.name}</span>
                  </div>
                  <button
                    onClick={() => removeFile(file.name)}
                    className="text-gray-400 hover:text-red-500"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Output Selection */}
        <div className="mt-8">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Select Output Format
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {outputTypes.map(({ id, name, icon: Icon }) => (
              <button
                key={id}
                onClick={() => toggleOutput(id)}
                className={`p-4 rounded-lg border-2 transition-colors flex flex-col items-center justify-center ${
                  selectedOutput === id
                    ? "border-[#382D76] bg-[#382D76]/10"
                    : "border-gray-200 hover:border-[#382D76]"
                }`}
              >
                <Icon className="h-6 w-6 text-gray-600" />
                <span className="mt-2 text-sm font-medium text-gray-900">
                  {name}
                </span>
                {selectedOutput === id && (
                  <CheckCircle className="mt-2 h-4 w-4 text-[#382D76]" />
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Additional Configuration Options */}
        {selectedOutput === "presentation" && (
          <div className="mt-6 p-6 bg-gray-50 rounded-lg border border-gray-200 shadow-sm">
            <h4 className="text-xl font-bold text-[#382D76] mb-4">
              Configure Presentation
            </h4>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Research Title
              </label>
              <input
                type="text"
                value={researchTitle}
                onChange={(e) => setResearchTitle(e.target.value)}
                placeholder="Enter Research Title"
                className="w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-[#382D76]"
              />
            </div>
            <p className="text-sm text-gray-600 mb-4">Select a style:</p>
            <div className="flex space-x-8">
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="radio"
                  name="presentationStyle"
                  value="professional"
                  checked={presentationStyle === "professional"}
                  onChange={(e) =>
                    setPresentationStyle(
                      e.target.value as "professional" | "fun"
                    )
                  }
                  className="form-radio text-[#382D76] h-5 w-5"
                />
                <span className="text-gray-700">Professional</span>
              </label>
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="radio"
                  name="presentationStyle"
                  value="fun"
                  checked={presentationStyle === "fun"}
                  onChange={(e) =>
                    setPresentationStyle(
                      e.target.value as "professional" | "fun"
                    )
                  }
                  className="form-radio text-[#382D76] h-5 w-5"
                />
                <span className="text-gray-700">Fun</span>
              </label>
            </div>
            <div className="mt-4">
              <button
                onClick={handleTransform}
                disabled={processing}
                className={`w-full px-4 py-3 rounded-lg text-white font-medium transition-colors ${
                  processing
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-[#382D76] hover:bg-[#382D76]/90"
                }`}
              >
                {processing ? "Transforming..." : "Generate Presentation"}
              </button>
            </div>
          </div>
        )}

        {selectedOutput === "podcast" && (
          <div className="mt-6 p-6 bg-gray-50 rounded-lg border border-gray-200 shadow-sm">
            <h4 className="text-xl font-bold text-[#382D76] mb-4">
              Configure Podcast
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label className="block mb-2 text-sm font-medium text-gray-700">
                  Length
                </label>
                <select
                  value={podcastLength}
                  onChange={(e) => setPodcastLength(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-[#382D76]"
                >
                  {lengthOptions.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block mb-2 text-sm font-medium text-gray-700">
                  Tone
                </label>
                <select
                  value={podcastTone}
                  onChange={(e) => setPodcastTone(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-[#382D76]"
                >
                  {toneOptions.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block mb-2 text-sm font-medium text-gray-700">
                  Language
                </label>
                <select
                  value={podcastLanguage}
                  onChange={(e) => setPodcastLanguage(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-[#382D76]"
                >
                  {languageOptions.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="mt-4">
              <button
                onClick={handleTransform}
                disabled={processing}
                className={`w-full px-4 py-3 rounded-lg text-white font-medium transition-colors ${
                  processing
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-[#382D76] hover:bg-[#382D76]/90"
                }`}
              >
                {processing ? "Transforming..." : "Generate Podcast"}
              </button>
            </div>
          </div>
        )}

        {selectedOutput === "video" && (
          <div className="mt-6 p-6 bg-gray-50 rounded-lg border border-gray-200 shadow-sm">
            
            <div className="mt-4">
              <button
                onClick={handleTransform}
                disabled={processing}
                className="w-full px-4 py-3 rounded-lg text-white font-medium transition-colors bg-[#382D76] hover:bg-[#382D76]/90"
              >
                {processing ? "Transforming..." : "Generate Video"}
              </button>
            </div>
          </div>
        )}
        {selectedOutput === "comic" && (
          <div className="mt-6 p-6 bg-gray-50 rounded-lg border border-gray-200 shadow-sm">
            
            <div className="mt-4">
              <button
                onClick={handleTransform}
                disabled={processing}
                className="w-full px-4 py-3 rounded-lg text-white font-medium transition-colors bg-[#382D76] hover:bg-[#382D76]/90"
              >
                {processing ? "Transforming..." : "Generate Comic"}
              </button>
            </div>
          </div>
        )}

        {/* Display results with both preview and download options */}
        {selectedOutput === "presentation" && pptUrl && (
          <div className="mt-4">
            <iframe
              src={pptUrl}
              title="Presentation Preview"
              className="w-full h-96 rounded-lg shadow-sm mb-4"
            />
            <a href={pptUrl} download target="_blank" rel="noopener noreferrer">
              <button className="w-full px-4 py-3 rounded-lg text-white font-medium bg-[#382D76] hover:bg-[#382D76]/90">
                Download Presentation
              </button>
            </a>
          </div>
        )}
        {selectedOutput === "podcast" && audioUrl && (
          <div className="mt-4">
            <audio controls src={audioUrl} className="w-full mb-4" />
            <a
              href={audioUrl}
              download
              target="_blank"
              rel="noopener noreferrer"
            >
              <button className="w-full px-4 py-3 rounded-lg text-white font-medium bg-[#382D76] hover:bg-[#382D76]/90">
                Download Podcast
              </button>
            </a>
          </div>
        )}
        {selectedOutput === "video" && videoUrl && (
          <div className="mt-4">
            <video controls src={videoUrl} className="w-full mb-4" />
            <a
              href={videoUrl}
              download
              target="_blank"
              rel="noopener noreferrer"
            >
              <button className="w-full px-4 py-3 rounded-lg text-white font-medium bg-[#382D76] hover:bg-[#382D76]/90">
                Download Video
              </button>
            </a>
          </div>
        )}
        {selectedOutput === "comic" && comicUrl && (
          <div className="mt-4">
            <img
              src={comicUrl}
              alt="Generated Comic"
              className="w-full rounded-lg shadow-sm mb-4"
            />
            <a
              href={comicUrl}
              download
              target="_blank"
              rel="noopener noreferrer"
            >
              <button className="w-full px-4 py-3 rounded-lg text-white font-medium bg-[#382D76] hover:bg-[#382D76]/90">
                Download Comic
              </button>
            </a>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResearchPage;
