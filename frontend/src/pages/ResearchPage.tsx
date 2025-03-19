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
  const ngrok_link = "https://8204-34-148-56-74.ngrok-free.app";
  // File upload & progress states
  const [files, setFiles] = React.useState<File[]>([]);
  const [uploadProgress, setUploadProgress] = React.useState(0);

  // Which output format is selected ("presentation", "podcast", "video", "comic")
  const [selectedOutput, setSelectedOutput] = React.useState<string>("none");
  // Global processing state
  const [processing, setProcessing] = React.useState(false);

  // Presentation-specific states
  const [researchTitle, setResearchTitle] = React.useState("");
  const [presentationStyle, setPresentationStyle] = React.useState<"professional" | "fun">("professional");
  // Storing the generated PDF as a blob URL
  const [pdfUrl, setPdfUrl] = React.useState<string | null>(null);

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
  const [podcastLength, setPodcastLength] = React.useState<string>(lengthOptions[0]);
  const [podcastTone, setPodcastTone] = React.useState<string>(toneOptions[0]);
  const [podcastLanguage, setPodcastLanguage] = React.useState<string>(languageOptions[0]);
  // Podcast audio URL (Object URL from the returned .wav Blob)
  const [audioUrl, setAudioUrl] = React.useState<string | null>(null);

  // Video-specific state
  const [videoUrl, setVideoUrl] = React.useState<string | null>(null);

  // Comic-specific states
  const [comicUrl, setComicUrl] = React.useState<string | null>(null);

  // Define available output types
  const outputTypes = [
    { id: "presentation", name: "Presentation", icon: Presentation },
    { id: "podcast", name: "Podcast", icon: Mic },
    { id: "video", name: "Video", icon: Video },
    { id: "comic", name: "Comic", icon: Image },
  ];

  // Simulate file upload progress
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

  // File drop handling using react-dropzone
  const onDrop = React.useCallback((acceptedFiles: File[]) => {
    setFiles(acceptedFiles);
    simulateUpload();
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
  });

  // Toggle output format selection (clearing previous results)
  const toggleOutput = (id: string) => {
    setSelectedOutput((prev) => (prev === id ? "none" : id));
    setPdfUrl(null);
    setAudioUrl(null);
    setVideoUrl(null);
    setComicUrl(null);
  };

  // Remove a file by name
  const removeFile = (name: string) => {
    setFiles((prev) => prev.filter((file) => file.name !== name));
  };

  // For Presentation: handle PDF output generation
  const handleGeneratePDF = async () => {
    if (!files[0] || !researchTitle) {
      alert("Please upload a PDF and enter a research title.");
      return;
    }
    setProcessing(true);
    const formData = new FormData();
    formData.append("file", files[0]);
    formData.append("presentation_title", researchTitle);
    const toneValue = presentationStyle === "professional" ? "formal" : "funny";
    formData.append("tone", toneValue);
    formData.append("ppt_type", "1"); // Default ppt type

    try {
      const response = await axios.post(
        `${ngrok_link}/generate_pdf`,
        formData,
        { responseType: "blob" }
      );
      const pdfBlob = new Blob([response.data], { type: "application/pdf" });
      const pdfBlobUrl = URL.createObjectURL(pdfBlob);
      setPdfUrl(pdfBlobUrl);
    } catch (error) {
      console.error("Error generating PDF:", error);
      alert("Failed to generate the PDF.");
    } finally {
      setProcessing(false);
    }
  };

  // For Podcast: handle podcast generation from PDF (backend now returns a .wav file)
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
      const response = await fetch( `${ngrok_link}/generate_podcast`, {
        method: "POST",
        body: formData,
      });
      if (response.ok) {
        // Get the .wav file as a Blob and create an Object URL for it
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        setAudioUrl(audioUrl);
      } else {
        console.error("Error processing podcast:", response.statusText);
        alert("Failed to process podcast.");
      }
    } catch (error) {
      console.error("Error processing podcast:", error);
    }
    setProcessing(false);
  };

  // For Video (Shorts)
  const handleShortsProcess = async () => {
    if (!files[0]) return;
    setProcessing(true);
    const formData = new FormData();
    formData.append("file", files[0]);

    try {
      const response = await fetch(`${ngrok_link}/generate_video`, {
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
      const response = await fetch("http://127.0.0.1:7088/predict", {
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

  // Unified handler that calls the appropriate function based on selected output
  const handleTransform = async () => {
    if (selectedOutput === "presentation") {
      await handleGeneratePDF();
    } else if (selectedOutput === "podcast") {
      await handlePodcastProcess();
    } else if (selectedOutput === "video") {
      await handleShortsProcess();
    } else if (selectedOutput === "comic") {
      await handleComicProcess();
    }
  };

  // Determine if an output has been generated.
  const isOutputReady =
    (selectedOutput === "presentation" && pdfUrl) ||
    (selectedOutput === "podcast" && audioUrl) ||
    (selectedOutput === "video" && videoUrl) ||
    (selectedOutput === "comic" && comicUrl);

  // If a PDF is ready, show the two-panel PDF layout.
  if (selectedOutput === "presentation" && pdfUrl) {
    return (
      <div className="w-[100vw] mx-auto px-4 py-8 flex justify-center">
        <div style={{ display: "flex", gap: "1rem", height: "80vh", width: "95vw" }}>
          {/* Left Panel: PDF Display */}
          <div style={{ flex: 2, height: "100%" }}>
            <iframe
              src={pdfUrl}
              title="Generated PDF"
              style={{
                height: "100%",
                border: "none",
                borderRadius: "8px",
                width: "70vw",
              }}
            />
          </div>
          {/* Right Panel: Notepad & Action Buttons */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
            <div
              style={{
                marginBottom: "0.5rem",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "0.5rem",
                border: "2px solid #382D76",
                borderRadius: "8px",
                fontWeight: "bold",
                color: "#382D76",
              }}
            >
              <span>Notepad 📝</span>
              <button
                onClick={() => {
                  /* Implement copy functionality */
                }}
                style={{
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                }}
              >
                Copy 🗒️
              </button>
            </div>
            <textarea
              placeholder="Take your notes here..."
              style={{
                flex: 1,
                borderRadius: "8px",
                padding: "0.5rem",
                border: "1px solid #ccc",
              }}
            />
            <div style={{ display: "flex", gap: "0.5rem", marginTop: "1rem" }}>
              <button
                onClick={() => window.print()}
                style={{
                  flex: 1,
                  backgroundColor: "#382D76",
                  color: "white",
                  padding: "0.5rem",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                }}
              >
                Print
              </button>
              <a
                href={pdfUrl}
                download="output.pdf"
                style={{
                  flex: 1,
                  backgroundColor: "#382D76",
                  color: "white",
                  padding: "0.5rem",
                  textAlign: "center",
                  borderRadius: "4px",
                  textDecoration: "none",
                }}
              >
                Download PDF
              </a>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // If a video is ready, show a similar two-panel layout with the video on the left.
  if (selectedOutput === "video" && videoUrl) {
    return (
      <div className="w-[100vw] mx-auto px-4 py-8 flex justify-center">
        <div style={{ display: "flex", gap: "1rem", height: "80vh", width: "95vw" }}>
          {/* Left Panel: Video Display */}
          <div style={{ flex: 2, height: "100%" }}>
            <video
              src={videoUrl}
              controls
              style={{
                height: "100%",
                border: "none",
                borderRadius: "8px",
                width: "70vw",
              }}
            />
          </div>
          {/* Right Panel: Notepad & Action Buttons */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
            <div
              style={{
                marginBottom: "0.5rem",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "0.5rem",
                border: "2px solid #382D76",
                borderRadius: "8px",
                fontWeight: "bold",
                color: "#382D76",
              }}
            >
              <span>Notepad 📝</span>
              <button
                onClick={() => {
                  /* Implement copy functionality */
                }}
                style={{
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                }}
              >
                Copy 🗒️
              </button>
            </div>
            <textarea
              placeholder="Take your notes here..."
              style={{
                flex: 1,
                borderRadius: "8px",
                padding: "0.5rem",
                border: "1px solid #ccc",
              }}
            />
            <div style={{ display: "flex", gap: "0.5rem", marginTop: "1rem" }}>
              <button
                onClick={() => window.print()}
                style={{
                  flex: 1,
                  backgroundColor: "#382D76",
                  color: "white",
                  padding: "0.5rem",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                }}
              >
                Print
              </button>
              <a
                href={videoUrl}
                download="final_output.mp4"
                style={{
                  flex: 1,
                  backgroundColor: "#382D76",
                  color: "white",
                  padding: "0.5rem",
                  textAlign: "center",
                  borderRadius: "4px",
                  textDecoration: "none",
                }}
              >
                Download Video
              </a>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // If a podcast is ready, show a two-panel layout with podcast audio and a blank notepad for notes.
  if (selectedOutput === "podcast" && audioUrl) {
    return (
      <div className="w-[100vw] mx-auto px-4 py-8 flex justify-center">
        <div style={{ display: "flex", gap: "1rem", height: "80vh", width: "95vw" }}>
          {/* Left Panel: Podcast Audio Display */}
          <div style={{ flex: 2, height: "100%" }}>
            <img
              className="w-full h-[63vh] mb-4 bg-black rounded-lg shadow-lg"
              src="https://res.cloudinary.com/dogfmhpfc/image/upload/v1740393379/DALL_E_2025-02-24_16.05.23_-_A_modern_podcast_studio_with_a_white_and_purple_theme._A_male_researcher_with_neatly_styled_white_hair_and_square_glasses_sits_at_a_table_engaged_in_c_jczglq.webp"
            ></img>
            <audio controls src={audioUrl} className="w-full" />
            <a
              href={audioUrl}
              download="podcast.wav"
              style={{
                display: "block",
                marginTop: "1rem",
                backgroundColor: "#382D76",
                color: "white",
                padding: "0.5rem",
                textAlign: "center",
                borderRadius: "4px",
                textDecoration: "none",
              }}
            >
              Download Podcast
            </a>
          </div>
          {/* Right Panel: Notepad for Personal Notes */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
            <div
              style={{
                marginBottom: "0.5rem",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "0.5rem",
                border: "2px solid #382D76",
                borderRadius: "8px",
                fontWeight: "bold",
                color: "#382D76",
              }}
            >
              <span>Notepad 📝</span>
              <button
                onClick={() => {
                  /* Implement copy functionality if needed */
                }}
                style={{
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                }}
              >
                Copy Notes 🗒️
              </button>
            </div>
            <textarea
              placeholder="Write your personal notes here..."
              style={{
                flex: 1,
                borderRadius: "8px",
                padding: "0.5rem",
                border: "1px solid #ccc",
              }}
            />
          </div>
        </div>
      </div>
    );
  }

  // If a comic is ready, show the comic result.
  if (selectedOutput === "comic" && comicUrl) {
    return (
      <div className="mt-4">
        <img
          src={comicUrl}
          alt="Generated Comic"
          className="w-full rounded-lg shadow-sm mb-4"
        />
        <a href={comicUrl} download target="_blank" rel="noopener noreferrer">
          <button className="w-full px-4 py-3 rounded-lg text-white font-medium bg-[#382D76] hover:bg-[#382D76]/90">
            Download Comic
          </button>
        </a>
      </div>
    );
  }

  // Otherwise, show the initial UI (upload, file list, output selection, configuration)
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

        {/* Upload Progress */}
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

        {/* Additional Configuration for Presentation - now with just a submit button */}
        {selectedOutput === "presentation" && (
          <div className="mt-6 p-6 bg-gray-50 rounded-lg border border-gray-200 shadow-sm">
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
        )}

        {/* Additional Configuration for Podcast - now with just a submit button */}
        {selectedOutput === "podcast" && (
          <div className="mt-6 p-6 bg-gray-50 rounded-lg border border-gray-200 shadow-sm">
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
        )}

        {/* Additional Configuration for Video */}
        {selectedOutput === "video" && (
          <div className="mt-6 p-6 bg-gray-50 rounded-lg border border-gray-200 shadow-sm">
            <div className="mt-4">
              <button
                onClick={handleTransform}
                disabled={processing}
                className="w-full px-4 py-3 rounded-lg text-white font-medium transition-colors bg-[#382D76] hover:bg-[#382D76]/90"
              >
                {processing ? "Transforming... , please wait it may take 3-4 mins " : "Generate Video"}
              </button>
            </div>
          </div>
        )}

        {/* Additional Configuration for Comic */}
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
      </div>
    </div>
  );
};

export default ResearchPage;
