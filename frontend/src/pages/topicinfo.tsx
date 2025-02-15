import React, { useState } from 'react';

interface MaterialItem {
  link: string;
  title: string;
}

interface StudyMaterial {
  pdf?: MaterialItem[];
  video?: MaterialItem[];
}

interface Topic {
  // Example structure:
  // topic.study_material = [
  //   { pdf: [{ link: "...", title: "..." }], video: [{ link: "...", title: "..." }] }
  // ]
  study_material: StudyMaterial[];
}

interface TopicPageProps {
  topic: Topic;
}

const TopicPage: React.FC<TopicPageProps> = ({ topic }) => {
  // Grab the first study_material object (adjust as needed)
  const study_material = topic.study_material[0];

  // State to store the currently selected material
  const [selectedMaterial, setSelectedMaterial] = useState<
    (MaterialItem & { type: 'pdf' | 'video' }) | null
  >(null);

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

          {/* Content Area: Two Panels (Left: Material, Right: Notepad & Back Button) */}
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

            {/* Right Panel: Notepad & Back Button */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              <textarea
                placeholder="Notepad..."
                style={{
                  flex: 1,
                  width: '100%',
                  padding: '0.5rem',
                  border: '1px solid #ccc',
                  borderRadius: '4px',
                  resize: 'none',
                }}
              />
              <button
                onClick={() => setSelectedMaterial(null)}
                style={{
                  marginTop: '1rem',
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
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {items.map((item, idx) => (
              <div
                key={idx}
                onClick={() => setSelectedMaterial(item)}
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
                  (e.currentTarget as HTMLDivElement).style.transform = 'translateY(0)';
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
    </div>
  );
};

export default TopicPage;
