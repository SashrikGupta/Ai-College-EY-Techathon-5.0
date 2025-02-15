import React, { useContext } from 'react';
import { Send } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { CurrConfigContext } from '../context.tsx';

const AdvisorPage = () => {
  const cont = useContext(CurrConfigContext) || {};

  console.log(cont.user?._id);

  const userid = cont.user?._id;
  const [message, setMessage] = React.useState('');
  const [messages, setMessages] = React.useState([
    { id: 1, text: "Hello! I'm your **AI Advisor**. How can I help you today?", sender: 'ai' },
  ]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim()) return;

    const userMessage = { id: messages.length + 1, text: message, sender: 'user' };
    const processingMessage = { id: messages.length + 2, text: '', sender: 'ai', loading: true };

    setMessages([...messages, userMessage, processingMessage]);
    setMessage('');

    try {
      const response = await fetch('http://localhost:5000/dbchat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userid, prompt: message }),
      });

      const data = await response.json();

      if (response.ok) {
        setMessages((prev) => [
          ...prev.slice(0, -1),
          { id: prev.length + 1, text: data.response, sender: 'ai' },
        ]);
      } else {
        throw new Error(data.error || 'Something went wrong');
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { id: prev.length + 1, text: 'Error: Unable to fetch response.', sender: 'ai' },
      ]);
      console.error('Error fetching response:', error);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="p-4 bg-primary-500 text-white">
          <h2 className="text-xl font-semibold">AI Advisor</h2>
          <p className="text-sm opacity-90">Get personalized guidance and support</p>
        </div>

        <div className="h-[500px] flex flex-col">
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-3 ${
                    msg.sender === 'user'
                      ? 'bg-primary-500 text-white'
                      : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  {msg.loading ? (
                    <div className="flex items-center space-x-2">
                      <span className="loader"></span>
                    </div>
                  ) : (
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  )}
                </div>
              </div>
            ))}
          </div>

          <form onSubmit={handleSubmit} className="p-4 border-t">
            <div className="flex space-x-4">
              <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Type your message..."
                className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
              <button
                type="submit"
                className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
              >
                <Send className="h-5 w-5" />
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default AdvisorPage;
