import React, { useState, useContext } from "react";
import {
  Briefcase,
  GraduationCap,
  ExternalLink,
  Upload,
} from "lucide-react";
import { CurrConfigContext } from "../context.tsx";

const IIICellPage = () => {
  const cont = useContext(CurrConfigContext) || {};
  const [role, setRole] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [jobs, setJobs] = useState<any[]>([]);
  const [courses, setCourses] = useState<any[]>([]);
  const [suggestion, setSuggestion] = useState("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  // Fetch jobs and courses based on the entered role
  const handleSubmit = async () => {
    if (!role) {
      alert("Please enter a job role.");
      return;
    }
    setLoading(true);

    const formData = new FormData();
    formData.append(
      "data",
      JSON.stringify({
        userid: cont?.user?._id,
        prompt: role,
      })
    );
    if (file) {
      formData.append("file", file);
    }

    try {
      const response = await fetch("http://127.0.0.1:5000/placement/search", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setJobs(data.Job_Listings || []);
        setCourses(data.Courses || []);
        setSuggestion(data.suggestion || "");
      } else {
        alert(data.error || "Something went wrong.");
      }
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  // Open the interview page in a new tab using query parameters
  const openInterview = (job: any) => {
    const url = `/interview?jobTitle=${encodeURIComponent(
      job.Job_Title
    )}&skills=${encodeURIComponent(job.Skills)}`;
    window.open(url, "_blank");
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-100 to-white py-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="space-y-12">
          {/* Header */}
          <header className="text-center">
            <h1 className="text-4xl font-bold text-[#382D76]">
              Career Guidance Dashboard
            </h1>
            <p className="mt-2 text-lg text-[#382D76]-300">
              Discover the perfect jobs and courses tailored just for you.
            </p>
          </header>

          {/* Initial Form */}
          {!jobs[0] && (
            <div className="bg-white rounded-xl shadow-xl p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">
                Get Started
              </h2>
              <div className="max-w-xl mx-auto">
                <label className="block text-sm font-medium text-gray-700">
                  What role are you interested in?
                </label>
                <input
                  type="text"
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  className="mt-2 block w-full rounded-md border border-gray-300 px-4 py-2 shadow-sm focus:border-indigo-500 focus:ring focus:ring-indigo-200"
                  placeholder="e.g., AI Engineer, Data Scientist"
                />
              </div>
              {/* File Upload */}
              <div className="mt-6 max-w-xl mx-auto">
                <label className="block text-sm font-medium text-gray-700">
                  Upload Placement Report (optional)
                </label>
                <div className="mt-2 flex items-center gap-3">
                  <input
                    type="file"
                    onChange={handleFileChange}
                    className="hidden"
                    id="upload-file"
                  />
                  <label
                    htmlFor="upload-file"
                    className="cursor-pointer flex items-center rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50"
                  >
                    <Upload className="h-5 w-5 mr-2" />
                    Choose File
                  </label>
                  {file && (
                    <span className="text-sm text-gray-500">{file.name}</span>
                  )}
                </div>
              </div>
              <div className="mt-8 text-center">
                <button
                  onClick={handleSubmit}
                  className="inline-block bg-[#382D76] text-white px-6 py-3 rounded-md shadow-lg hover:bg-[#312465] focus:outline-none focus:ring-2 focus:ring-[#382D76] focus:ring-offset-2 transition-colors duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                  disabled={loading}
                >
                  {loading ? "Searching..." : "Find Jobs & Courses"}
                </button>
              </div>
            </div>
          )}

          {/* Results */}
          <div className="space-y-12">
            {/* AI Suggestion */}
            {suggestion && (
              <div className="bg-indigo-50 border-l-4 border-indigo-500 rounded-xl shadow-xl p-8">
                <h3 className="text-2xl font-semibold text-[#382D76] mb-4">
                  AI Career Advice
                </h3>
                <p className="text-gray-700 leading-relaxed">{suggestion}</p>
              </div>
            )}

            {/* Job Listings */}
            {jobs.length > 0 && (
              <div className="bg-white rounded-xl shadow-xl p-8">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-2xl font-semibold text-[#382D76]">
                    Relevant Job Openings
                  </h3>
                  <Briefcase className="h-6 w-6 text-gray-500" />
                </div>
                <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                  {jobs.map((job, index) => (
                    <div
                      key={index}
                      className="bg-gray-50 rounded-lg p-6 hover:shadow-lg transition-shadow duration-200"
                    >
                      <h4 className="font-semibold text-gray-900 text-lg">
                        {job.Job_Title}
                      </h4>
                      <p className="text-sm text-gray-600 mt-1">
                        {job.Company}
                      </p>
                      <p className="text-sm text-gray-500 mt-1">
                        Skills: {job.Skills}
                      </p>
                      <div className="flex items-center gap-4 mt-4">
                        <a
                          href={job.Link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center text-sm text-indigo-600 hover:text-indigo-500"
                        >
                          Apply Now
                          <ExternalLink className="ml-2 h-4 w-4" />
                        </a>
                        <button
                          onClick={() => openInterview(job)}
                          className="text-sm text-[#382D76] font-semibold hover:underline"
                        >
                          Take Interview
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommended Courses */}
            {courses.length > 0 && (
              <div className="bg-white rounded-xl shadow-xl p-8">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-2xl font-semibold text-[#382D76]">
                    Recommended Courses
                  </h3>
                  <GraduationCap className="h-6 w-6 text-gray-500" />
                </div>
                <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                  {courses.map((course, index) => (
                    <div
                      key={index}
                      className="bg-gray-50 rounded-lg p-6 hover:shadow-lg transition-shadow duration-200"
                    >
                      <h4 className="font-semibold text-gray-900 text-lg">
                        {course.Course_Title}
                      </h4>
                      <p className="text-sm text-gray-600 mt-1">
                        {course.Snippet}
                      </p>
                      <a
                        href={course.Link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="mt-4 inline-flex items-center text-sm text-indigo-600 hover:text-indigo-500"
                      >
                        Learn More
                        <ExternalLink className="ml-2 h-4 w-4" />
                      </a>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default IIICellPage;
