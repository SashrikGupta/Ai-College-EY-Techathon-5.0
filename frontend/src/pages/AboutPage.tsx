import React from 'react';
import { Link } from 'react-router-dom';
import { BrainCircuit, BookOpen, GraduationCap, Users, FileText, Play } from 'lucide-react';

const AboutPage = () => {
  return (
    <div className="min-h-screen bg-white">
      {/* Hero Section */}
      <div className="relative py-16 bg-gradient-to-b from-primary-50 to-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:grid lg:grid-cols-12 lg:gap-8">
            <div className="sm:text-center md:max-w-2xl md:mx-auto lg:col-span-6 lg:text-left">
              <h1>
                <span className="block text-base text-primary-600 font-semibold tracking-wide uppercase">About Us</span>
                <span className="mt-1 block text-4xl tracking-tight font-extrabold sm:text-5xl xl:text-6xl">
                  <span className="block text-gray-900">Transforming Education</span>
                  <span className="block text-primary-600">With AI Technology</span>
                </span>
              </h1>
              <p className="mt-3 text-base text-gray-500 sm:mt-5 sm:text-xl lg:text-lg xl:text-xl">
                Our platform combines cutting-edge artificial intelligence with educational expertise to deliver personalized learning experiences, career guidance, and research transformation tools.
              </p>
            </div>
            <div className="mt-12 relative sm:max-w-lg sm:mx-auto lg:mt-0 lg:max-w-none lg:mx-0 lg:col-span-6 lg:flex lg:items-center">
              <div className="relative mx-auto w-full rounded-lg shadow-lg lg:max-w-md">
                <button
                  type="button"
                  className="relative block w-full bg-white rounded-lg overflow-hidden focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
                >
                  <video
                    className="w-full"
                    src="https://www.youtube.com/watch?v=MFnn2zj3byA"
                    
                  />
                  <div className="absolute inset-0 w-full h-full flex items-center justify-center">
                    <div className="flex items-center justify-center h-20 w-20 rounded-full bg-primary-500 bg-opacity-75">
                      <Play className="h-8 w-8 text-white" />
                    </div>
                  </div>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Section */}
      <div className="py-16 bg-gray-50 overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 space-y-8 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-base text-primary-600 font-semibold tracking-wide uppercase">Features</h2>
            <p className="mt-2 text-3xl font-extrabold text-gray-900 sm:text-4xl">Everything you need to succeed</p>
            <p className="mt-4 max-w-2xl mx-auto text-xl text-gray-500">
              Discover how our AI-powered platform can enhance your educational journey
            </p>
          </div>

          <div className="mt-16">
            <div className="space-y-12 lg:space-y-0 lg:grid lg:grid-cols-3 lg:gap-x-8">
              <div className="relative p-8 bg-white rounded-2xl shadow-lg">
                <div className="absolute top-8 left-8">
                  <BrainCircuit className="h-12 w-12 text-primary-500" />
                </div>
                <div className="mt-16 pt-4">
                  <h3 className="text-xl font-medium text-gray-900">AI Advisor</h3>
                  <p className="mt-4 text-base text-gray-500">
                    Get personalized academic guidance and career advice from our advanced AI system.
                  </p>
                </div>
              </div>

              <div className="relative p-8 bg-white rounded-2xl shadow-lg">
                <div className="absolute top-8 left-8">
                  <BookOpen className="h-12 w-12 text-primary-500" />
                </div>
                <div className="mt-16 pt-4">
                  <h3 className="text-xl font-medium text-gray-900">Smart Librarian</h3>
                  <p className="mt-4 text-base text-gray-500">
                    Interactive Q&A with your study materials and video content for enhanced learning.
                  </p>
                </div>
              </div>

              <div className="relative p-8 bg-white rounded-2xl shadow-lg">
                <div className="absolute top-8 left-8">
                  <FileText className="h-12 w-12 text-primary-500" />
                </div>
                <div className="mt-16 pt-4">
                  <h3 className="text-xl font-medium text-gray-900">ResearchGen</h3>
                  <p className="mt-4 text-base text-gray-500">
                    Transform research papers into engaging multimedia content automatically.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-12 text-center">
            <Link
              to="/login"
              className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700"
            >
              Get Started
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutPage;