import React, { useState, useEffect } from 'react';
import { ToastContainer, toast } from 'react-toastify'; // import toast
import 'react-toastify/dist/ReactToastify.css'; // import styles
import ReactMarkdown from 'react-markdown';
interface MaterialItem {
  link: string;
  title: string;
  notes?: string;
}

interface StudyMaterial {
  pdf?: MaterialItem[];
  video?: MaterialItem[];
}

interface Topic {
  study_material: StudyMaterial[];
}

interface TopicPageProps {
  topic: Topic;
  subcourseidx: number;
  topicidx: number;
  courseid: string;
}

const TopicPage: React.FC<TopicPageProps> = ({
  topic,
  subcourseidx,
  topicidx,
  courseid,
}) => {
  // Grab the first study_material object (adjust as needed)
  const study_material = topic.study_material[0];

  // State to store the currently selected material
  const [selectedMaterial, setSelectedMaterial] = useState<
    (MaterialItem & { type: 'pdf' | 'video' }) | null
  >(null);

  // Separate state for the notepad text
  const [notes, setNotes] = useState<string>('write your rough notes here and click ai refactor to enhance your notes ...');
  // sindex represents the study material index
  const [sindex, setsindex] = useState(0);

  // When a material is selected, load its notes (or default text) into the notepad state
  useEffect(() => {
    if (selectedMaterial) {
      setNotes(selectedMaterial.notes || 'Write your rough notes here and click Ai refactor to enhance your notes ...');
    }
  }, [selectedMaterial]);

  // Combine pdf and video items into one list with a 'type' property
  const items: (MaterialItem & { type: 'pdf' | 'video' })[] = [];
  if (study_material?.pdf) {
    study_material.pdf.forEach((pdfItem) => {
      items.push({ ...pdfItem, type: 'pdf' });
    });
  }
  if (study_material?.video) {
    study_material.video.forEach((videoItem) => {
      items.push({ ...videoItem, type: 'video' });
    });
  }

  // Detect if a URL is a YouTube link
  const isYouTubeLink = (url: string) =>
    url.includes('youtube.com') || url.includes('youtu.be');

  // Convert a YouTube watch URL to an embed URL
  const getYouTubeEmbedUrl = (url: string) => {
    return url.replace('watch?v=', 'embed/');
  };

  // Function to save notes to the server via API
  const saveNotes = async () => {
    const data = {
      courseid,
      subcourseidx,
      topicidx,
      study_index: sindex,
      notes_text: notes,
    };

    console.log(data);
  
    try {
      const response = await fetch('http://127.0.0.1:5000/update_notes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      // First read the response as text
      const text = await response.text();
      let json = {};
      if (text) {
        json = JSON.parse(text);
      }
  
      if (response.ok) {
        toast.success('Notes updated successfully!');
        // Optionally update the selected material's notes:
        if (selectedMaterial) {
          setSelectedMaterial({ ...selectedMaterial, notes });
        }
      } else {
        toast.error('Failed to update notes: ' + (json.error || 'Unknown error'));
      }
    } catch (error) {
      toast.error('Error updating notes: ' + error);
    }
  };

  // New function for AI Refactor using Llama model
  const handleAiRefactor = async () => {
    const apiKey = "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"; // ensure this is set in your environment
    const endpoint = "https://api.groq.com/openai/v1/chat/completions";
    const payload = {
      messages: [
        {
          role: "system",
          content:
            "Refactor the following notes into formal, short, and concise notes covering every point. Do not include anything else. Also do not use Markdown just plain text display in points "
        },
        {
          role: "user",
          content: notes
        }
      ],
      model: "llama-3.3-70b-versatile",
      temperature: 1,
      max_completion_tokens: 1024,
      top_p: 1,
      stream: false, // using synchronous response for simplicity
      stop: null
    };

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${apiKey}`
        },
        body: JSON.stringify(payload)
      });
      
      if (!response.ok) {
        throw new Error("Failed to call AI model");
      }
      const data = await response.json();
      // Assumes the API returns the refactored note in this path:
      const refinedNotes = data.choices[0].message.content.trim();
      setNotes(refinedNotes);
      // Optionally update the selected material's notes:
      if (selectedMaterial) {
        setSelectedMaterial({ ...selectedMaterial, notes: refinedNotes });
      }
      toast.success("Notes refactored successfully!");
    } catch (error: any) {
      toast.error("Error refactoring notes: " + error.message);
    }
  };

  if (!study_material || items.length === 0) {
    return <div>No study materials found.</div>;
  }

  return (
    <div
      style={{
        minHeight: '100vh',
        backgroundColor: '#fff',
        color: '#000',
        fontFamily: 'Arial, sans-serif',
      }}
    >
      {selectedMaterial ? (
        // Detail View
        <div style={{ padding: '1rem' }}>
          {/* Header: Title, Type, and Link */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '1rem',
            }}
          >
            <div>
              <h2 style={{ color: '#382D76', margin: 0 }}>
                {selectedMaterial.title}
              </h2>
              <span
                style={{
                  display: 'inline-block',
                  padding: '0.25rem 0.5rem',
                  backgroundColor: '#382D76',
                  color: '#fff',
                  borderRadius: '4px',
                  fontSize: '0.8rem',
                  marginTop: '0.5rem',
                }}
              >
                {selectedMaterial.type.toUpperCase()}
              </span>
            </div>
            <a
              href={selectedMaterial.link}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                fontSize: '1.5rem',
                textDecoration: 'none',
                color: '#382D76',
              }}
            >
              🔗
            </a>
          </div>

          {/* Content Area: Two Panels (Left: Material, Right: Notepad & Buttons) */}
          <div style={{ display: 'flex', gap: '1rem', height: '80vh' }}>
            {/* Left Panel: Material Display */}
            <div style={{ flex: 2, height: '100%' }}>
              {selectedMaterial.type === 'pdf' ? (
                <iframe
                  src={selectedMaterial.link}
                  style={{
                    width: '100%',
                    height: '100%',
                    border: 'none',
                    borderRadius: '8px',
                  }}
                  title={selectedMaterial.title}
                />
              ) : isYouTubeLink(selectedMaterial.link) ? (
                <iframe
                  width="100%"
                  height="100%"
                  src={getYouTubeEmbedUrl(selectedMaterial.link)}
                  frameBorder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                  title={selectedMaterial.title}
                  style={{ borderRadius: '8px' }}
                />
              ) : (
                <video
                  src={selectedMaterial.link}
                  controls
                  style={{
                    width: '100%',
                    height: '100%',
                    borderRadius: '8px',
                  }}
                />
              )}
            </div>

            {/* Right Panel: Notepad & Buttons */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              {/* NotePad Title with AI Refactor Button */}
              <div
                style={{
                  margin: '0 0 0.5rem',
                  textAlign: 'center',
                  fontWeight: 'bold',
                  color: 'black',
                  padding: '0.3rem',
                  border: '3px solid',
                  borderColor: '#382D76',
                }}
                className="rounded-lg flex justify-between"
              >
                <div>NotePad 📝</div>
                <button onClick={handleAiRefactor} className ="bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 bg-clip-text text-transparent">⚡Ai Refactor</button>
              </div>

              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                style={{
                  flex: 1,
                  width: '100%',
                  padding: '0.5rem',
                  border: '1px solid #ccc',
                  borderRadius: '4px',
                  resize: 'none',
                }}
              />
              <div
                style={{
                  display: 'flex',
                  gap: '1rem',
                  marginTop: '1rem',
                }}
              >
                <button
                  onClick={saveNotes}
                  style={{
                    flex: 1,
                    backgroundColor: 'green',
                    color: '#fff',
                    border: 'none',
                    padding: '0.5rem 1rem',
                    cursor: 'pointer',
                    borderRadius: '4px',
                  }}
                >
                  Save My Notes
                </button>
                <button
                  onClick={() => setSelectedMaterial(null)}
                  style={{
                    flex: 1,
                    backgroundColor: '#382D76',
                    color: '#fff',
                    border: 'none',
                    padding: '0.5rem 1rem',
                    cursor: 'pointer',
                    borderRadius: '4px',
                  }}
                >
                  Back to Study Materials
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : (
        // List View remains as before (linear layout with boxes)
        <div style={{ padding: '2rem' }}>
          <h2
            style={{
              color: '#382D76',
              textAlign: 'center',
              marginBottom: '2rem',
            }}
          >
            Study Materials
          </h2>
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '1.5rem',
            }}
          >
            {items.map((item, idx) => (
              <div
                key={idx}
                onClick={() => {
                  setSelectedMaterial(item);
                  setsindex(idx);
                }}
                style={{
                  backgroundColor: '#f9f9f9',
                  border: '2px solid #382D76',
                  borderRadius: '8px',
                  padding: '1.5rem',
                  cursor: 'pointer',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLDivElement).style.transform =
                    'translateY(-5px)';
                  (e.currentTarget as HTMLDivElement).style.boxShadow =
                    '0 4px 12px rgba(0, 0, 0, 0.2)';
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLDivElement).style.transform =
                    'translateY(0)';
                  (e.currentTarget as HTMLDivElement).style.boxShadow = 'none';
                }}
              >
                <h3 style={{ color: '#382D76', marginBottom: '0.5rem' }}>
                  {item.title}
                </h3>
                <span
                  style={{
                    display: 'inline-block',
                    padding: '0.25rem 0.5rem',
                    backgroundColor: '#382D76',
                    color: '#fff',
                    borderRadius: '4px',
                    fontSize: '0.8rem',
                  }}
                >
                  {item.type.toUpperCase()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      {/* Toast container to display toasts */}
      <ToastContainer position="top-right" autoClose={3000} />
    </div>
  );
};

export default TopicPage;
