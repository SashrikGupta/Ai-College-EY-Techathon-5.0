import React, { useState, useCallback, useEffect, useRef, useContext } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, File, X, MessageSquare } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { CurrConfigContext } from "../context.tsx";
import { useNavigate } from "react-router-dom";

const Chatbot = () => {
  const cont = useContext(CurrConfigContext) || {};
  const [messages, setMessages] = useState([
    { text: `Hi **${cont.user?.name}** ! how can I assist you?`, sender: "bot" }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesContainerRef = useRef(null);

  // Auto-scroll to the newest message whenever messages update or while loading
  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop =
        messagesContainerRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSendMessage = async () => {
    if (input.trim() === "") return;
    // Append user message
    setMessages((prev) => [...prev, { text: input, sender: "user" }]);
    const prompt = input;
    setInput("");

    setIsLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:5000/librarian/query_material", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ prompt })
      });

      if (!res.ok) {
        const errorData = await res.json();
        setMessages((prev) => [
          ...prev,
          { text: "Error: " + errorData.error, sender: "bot" }
        ]);
        setIsLoading(false);
        return;
      }

      const data = await res.json();
      // Handle response as string or as an object with a "message" property
      const botMessage =
        typeof data === "string" ? data : data.message || JSON.stringify(data);
      setMessages((prev) => [...prev, { text: botMessage, sender: "bot" }]);
    } catch (error) {
      console.error("Error sending query", error);
      setMessages((prev) => [
        ...prev,
        { text: "Error processing query", sender: "bot" }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full p-4 border shadow-md bg-white">
      <h3 className="text-lg font-semibold text-[#382D76] mb-2 flex items-center">
        <MessageSquare className="h-5 w-5 mr-2" /> AI Assistant
      </h3>
      <div
        ref={messagesContainerRef}
        className="flex-grow overflow-y-auto border p-2 rounded-md bg-gray-50"
      >
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`mb-2 text-sm ${
              msg.sender === "user" ? "text-right" : "text-left"
            }`}
          >
            <span
              className={`px-3 py-1 rounded-lg inline-block ${
                msg.sender === "user"
                  ? "bg-[#382D76] text-white"
                  : "bg-gray-300 text-gray-900"
              }`}
            >
              {msg.sender === "bot" ? (
                <ReactMarkdown className="prose prose-sm">
                  {msg.text}
                </ReactMarkdown>
              ) : (
                msg.text
              )}
            </span>
          </div>
        ))}
        {isLoading && (
          <div className="mb-2 text-left">
            <span className="loader px-3 py-1 rounded-lg inline-block bg-gray-300 text-gray-900">
              
            </span>
          </div>
        )}
      </div>
      <div className="mt-2 flex">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="flex-grow p-2 border rounded-l-md focus:outline-none"
          placeholder="Ask a question..."
          disabled={isLoading}
        />
        <button
          onClick={handleSendMessage}
          disabled={isLoading}
          className={`bg-[#382D76] text-white px-4 py-2 rounded-r-md hover:bg-indigo-700 ${
            isLoading ? "opacity-50 cursor-not-allowed" : ""
          }`}
        >
          {isLoading ? "Processing..." : "Send"}
        </button>
      </div>
    </div>
  );
};

const LibrarianPage = () => {
  const cont = useContext(CurrConfigContext) || {};
  const navigate = useNavigate();
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);
  // For the file upload UI (pre-chat) we keep the selected PDF preview...
  const [selectedPdf, setSelectedPdf] = useState(null);
  // For the chat view we use this state for the file preview dropdown
  const [selectedChatFile, setSelectedChatFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const onDrop = useCallback(
    (acceptedFiles) => {
      setFiles((prev) => [...prev, ...acceptedFiles]);
      // For the file upload view, if there's a PDF and none is selected yet, select the first one.
      const pdfFile = acceptedFiles.find(
        (file) => file.type === "application/pdf"
      );
      if (pdfFile && !selectedPdf) {
        setSelectedPdf(pdfFile);
      }
      // For the chat view, if none is selected yet, set the first file.
      if (!selectedChatFile && acceptedFiles.length > 0) {
        setSelectedChatFile(acceptedFiles[0]);
      }
    },
    [selectedPdf, selectedChatFile]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "video/*": [".mp4", ".avi", ".mov"]
    }
  });

  const removeFile = (name) => {
    setFiles((prev) => prev.filter((file) => file.name !== name));
    if (selectedPdf && selectedPdf.name === name) {
      setSelectedPdf(null);
    }
    if (selectedChatFile && selectedChatFile.name === name) {
      setSelectedChatFile(null);
    }
  };

  const handleUpload = async () => {
    setUploading(true);
    setIsProcessing(true); // Hide the upload box after processing starts

    const formData = new FormData();
    files.forEach((file) => {
      formData.append("files", file);
    });

    // Replace with a dynamic user ID as needed.
    const data = { userid: cont?.user?._id };
    formData.append("data", JSON.stringify(data));

    try {
      const res = await fetch("http://127.0.0.1:5000/librarian/add_material", {
        method: "POST",
        body: formData
      });
      if (!res.ok) {
        const errorData = await res.json();
        console.error("Upload error:", errorData);
        setUploading(false);
        return;
      }
      const jsonData = await res.json();
      console.log("Upload success:", jsonData);
      setUploading(false);
      setShowChatbot(true);
      // The files remain in state so they can be viewed in the chat view.
    } catch (error) {
      console.error("Error uploading files:", error);
      setUploading(false);
    }
  };

  // Auto-select the first file for chat preview if none is selected yet.
  useEffect(() => {
    if (!selectedChatFile && files.length > 0) {
      setSelectedChatFile(files[0]);
    }
  }, [files, selectedChatFile]);

  // Clear Memory function:
  const handleClearMemory = async () => {
    try {
      const res = await fetch("http://127.0.0.1:5000/librarian/delete", {
        method: "GET"
      });
      if (!res.ok) {
        const errorData = await res.json();
        console.error("Error deleting material:", errorData);
        return;
      }
      const data = await res.json();
      console.log("Material deleted successfully:", data);
      // Navigate to the profile page after successful deletion.
      navigate("/profile");
    } catch (error) {
      console.error("Error calling delete endpoint:", error);
    }
  };

  // --- Render Layout ---

  // If files have been processed, show the new layout with a left-side file preview dropdown and right-side Chatbot.
  if (showChatbot) {
    return (
      <div className="w-[100vw] mx-auto px-4 pt-8 mt-4 h-[85vh]">
        <div className="flex flex-col lg:flex-row justify-between">
          {/* Left Column: File Preview Dropdown */}
          <div className="h-[85vh] w-[70vw]">
            <div className="w-[70vw] flex justify-between">
              <select
                value={selectedChatFile ? selectedChatFile.name : ""}
                onChange={(e) => {
                  const file = files.find((f) => f.name === e.target.value);
                  setSelectedChatFile(file);
                }}
                className="w-[60vw] h-[5vh] p-2 border"
              >
                {files.map((file) => (
                  <option key={file.name} value={file.name}>
                    {file.name}
                  </option>
                ))}
              </select>
              <button
                onClick={handleClearMemory}
                className="w-[10vw] h-[5vh] font-bold flex justify-center items-center bg-[#382D76] text-white"
              >
                Clear Memory
              </button>
            </div>
            <div className="mb-[2vh] shadow-lg">
              {selectedChatFile && selectedChatFile.type === "application/pdf" ? (
                <iframe
                  src={URL.createObjectURL(selectedChatFile)}
                  title="File Preview"
                  className="w-full h-[80vh] border"
                />
              ) : selectedChatFile && selectedChatFile.type.startsWith("video/") ? (
                <video
                  src={URL.createObjectURL(selectedChatFile)}
                  controls
                  className="w-full h-64 border"
                />
              ) : (
                <div className="w-full h-64 flex items-center justify-center border">
                  <p className="text-gray-500">
                    Preview not available for this file type.
                  </p>
                </div>
              )}
            </div>
          </div>
          {/* Right Column: Chatbot */}
          <div className="h-[85vh] w-[28vw]">
            <Chatbot />
          </div>
        </div>
      </div>
    );
  }

  // --- Pre-Upload Layout (File Upload UI) ---
  return (
    <div className="max-w-6xl mx-auto px-4 py-8 bg-white mt-4">
      <h2 className="text-3xl font-bold text-[#382D76] mb-6 text-center">
        Smart Librarian
      </h2>
      <p className="text-gray-600 mb-8 text-center">
        Upload your study materials and get instant answers to your questions.
      </p>
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Left Column: File Upload & Preview */}
        <div className="lg:w-1/2 bg-white p-4 rounded-lg shadow-lg">
          {!isProcessing && (
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? "border-[#382D76] bg-gray-50"
                  : "border-gray-300 hover:border-[#382D76]"
              }`}
            >
              <input {...getInputProps()} />
              <Upload className="mx-auto h-12 w-12 text-gray-400" />
              <p className="mt-2 text-sm text-gray-600">
                Drag and drop your files here, or click to select files
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Supports PDF documents and video files
              </p>
            </div>
          )}
          {files.length > 0 && (
            <div className="mt-6">
              <h3 className="text-lg font-semibold text-[#382D76] mb-4">
                Selected Files
              </h3>
              {/* Scrollable container for the file list */}
              <div className="max-h-64 overflow-y-auto space-y-2">
                {files.map((file) => (
                  <div
                    key={file.name}
                    className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                  >
                    <div className="flex items-center">
                      <File className="h-5 w-5 text-gray-400 mr-2" />
                      <span className="text-sm text-gray-900 break-all">
                        {file.name}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {file.type === "application/pdf" && (
                        <button
                          onClick={() => setSelectedPdf(file)}
                          className="text-[#382D76] hover:underline text-sm"
                        >
                          View
                        </button>
                      )}
                      <button
                        onClick={() => removeFile(file.name)}
                        className="text-gray-400 hover:text-red-500"
                      >
                        <X className="h-5 w-5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
              <button
                onClick={handleUpload}
                disabled={uploading}
                className={`mt-4 w-full px-4 py-2 rounded-md text-white ${
                  uploading
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-[#382D76] hover:bg-indigo-700"
                }`}
              >
                {uploading ? "Processing..." : "Process Files"}
              </button>
            </div>
          )}
          {selectedPdf && (
            <div className="mt-6">
              <h3 className="text-lg font-semibold text-[#382D76] mb-2">
                PDF Preview: {selectedPdf.name}
              </h3>
              <iframe
                src={URL.createObjectURL(selectedPdf)}
                title="PDF Preview"
                className="w-full h-96 border"
              />
            </div>
          )}
        </div>
        {/* Right Column: Placeholder for Chatbot */}
        <div className="lg:w-1/2 h-full">
          <div className="flex items-center justify-center h-full bg-gray-100 rounded-lg shadow-lg p-4">
            <p className="text-gray-500">
              Chatbot will appear here after processing files.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LibrarianPage;
