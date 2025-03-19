import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2 } from 'lucide-react';
import { GoogleGenerativeAI } from "@google/generative-ai";
import ReactMarkdown from 'react-markdown';
import Stanford from "../courses.json";

// Initialize Gemini API
const genAI = new GoogleGenerativeAI("AIzaSyDOisJFtcZNIfGQPGQ1XiCO_uJ6i4TretI");

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

const ChatbotPage = ({ interest }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(true); // Start with loading state
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [model, setModel] = useState<any>(null);
  const [chat, setChat] = useState<any>(null);

  useEffect(() => {
    // Initialize the chat model with system prompt
    console.log(interest)
    const initChat = async () => {
      try {
        const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });
        
        // Create system prompt
        const systemPrompt = `You are a subject matter expert. Your task is to do the following:
        
        The user wants to study ${interest}. Your task is to:
        
        1. First educate the student about the course and what they will be studying
        2. Then ask whether they want an evaluation report
        3. If yes, ask 6-7 questions that judge the student on the prerequisites required (for example: "Do you have understanding of basic quantum physics? If yes, tell what is Bohr's model" or "Are you proficient in Python? If so, tell the function of enumerate")
        4. At the end, give a small evaluation report where you very briefly (4-5 lines) evaluate the student and what prerequisites might be necessary for them to study
        
        NOTE: In the final report, only mention prerequisites they are weak on. If they are good in a particular prerequisite, skip mentioning it.`;

        // Start chat with history configuration
        const chat = model.startChat({
          generationConfig: {
            temperature: 0.7,
            topK: 40,
            topP: 0.95,
            maxOutputTokens: 2048,
          },
          history: [],
          safetySettings: [
            {
              category: "HARM_CATEGORY_HARASSMENT",
              threshold: "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
              category: "HARM_CATEGORY_HATE_SPEECH",
              threshold: "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
              category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
              threshold: "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
              category: "HARM_CATEGORY_DANGEROUS_CONTENT",
              threshold: "BLOCK_MEDIUM_AND_ABOVE"
            }
          ],
        });
        
        setModel(model);
        setChat(chat);
        
        // Send initial system prompt as user message
        const result = await chat.sendMessage(systemPrompt);
        const response = await result.response;
        
        // Set initial bot message
        const welcomeMessage: Message = {
          id: Date.now().toString(),
          text: response.text(),
          sender: 'bot',
          timestamp: new Date(),
        };
        
        setMessages([welcomeMessage]);
        setIsLoading(false);
      } catch (error) {
        console.error('Error initializing chat:', error);
        // Show error message if initialization fails
        setMessages([{
          id: '1',
          text: 'Sorry, I had trouble connecting to the AI service. Please try again later.',
          sender: 'bot',
          timestamp: new Date(),
        }]);
        setIsLoading(false);
      }
    };
    
    initChat();
  }, [interest]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || !chat) return;

    const newMessage: Message = {
      id: Date.now().toString(),
      text: inputMessage,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, newMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Send message to Gemini and get response
      const result = await chat.sendMessage(inputMessage);
      const response = await result.response;
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: response.text(),
        sender: 'bot',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, botResponse]);
    } catch (error) {
      console.error('Error calling Gemini API:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'I apologize, but I encountered an error. Please try again.',
        sender: 'bot',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const MessageContent = ({ message }: { message: Message }) => {
    if (message.sender === 'user') {
      return <p className="text-sm whitespace-pre-wrap">{message.text}</p>;
    }
    
    return (
      <div className="prose prose-sm max-w-none prose-pre:bg-gray-800 prose-pre:text-white">
        <ReactMarkdown>{message.text}</ReactMarkdown>
      </div>
    );
  };

  return (
    <>
      {/* Chat Container */}
      <div className="overflow-hidden">
        {/* Messages Area */}
        <div className="h-[500px] overflow-y-auto p-6">
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex items-start gap-2.5 ${
                  message.sender === 'user' ? 'flex-row-reverse' : ''
                }`}
              >
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center ${
                    message.sender === 'user'
                      ? 'bg-primary-100 text-primary-600'
                      : 'bg-gray-100 text-gray-600'
                  }`}
                >
                  {message.sender === 'user' ? (
                    <User className="w-5 h-5" />
                  ) : (
                    <Bot className="w-5 h-5" />
                  )}
                </div>
                <div
                  className={`max-w-[75%] rounded-lg p-4 ${
                    message.sender === 'user'
                      ? 'bg-primary-500 text-white'
                      : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  <MessageContent message={message} />
                  <span className="text-xs mt-2 block opacity-70">
                    {message.timestamp.toLocaleTimeString()}
                  </span>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex items-center gap-2 text-gray-500">
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>AI is thinking...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <form
          onSubmit={handleSendMessage}
          className="border-t border-gray-200 p-4"
        >
          <div className="flex gap-4">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Type your message here..."
              className="flex-1 min-w-0 rounded-lg border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!inputMessage.trim() || isLoading}
              className="inline-flex items-center justify-center rounded-lg px-4 py-2 bg-primary-500 text-white hover:bg-primary-600 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
        </form>
      </div>
    </>
  );
};

export default ChatbotPage;