import React, { useRef, useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Loader2, Mic, MicOff, Video, VideoOff, PhoneOff, Send, ArrowRight, Volume2, VolumeX } from "lucide-react";
import ReactMarkdown from "react-markdown";

// Initialize Gemini API
const genAI = new GoogleGenerativeAI("AIzaSyDmtHHE-eXi0fTTAfDOr8-Uks4bZq5ffoA");

const InterviewPage = () => {
  const [searchParams] = useSearchParams();
  const jobTitle = searchParams.get("jobTitle") || "Software Engineer";
  const skills = searchParams.get("skills") || "JavaScript, React, Node.js";

  // State for video
  const [cameraOn, setCameraOn] = useState(false);
  const [localStream, setLocalStream] = useState(null);
  const videoRef = useRef(null);

  // State for audio/speech recognition
  const [isListening, setIsListening] = useState(false);
  const [userSpeech, setUserSpeech] = useState("");
  const [recognition, setRecognition] = useState(null);

  // State for interview questions and answers
  const [questions, setQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [model, setModel] = useState(null);
  const [chat, setChat] = useState(null);
  const [interviewComplete, setInterviewComplete] = useState(false);
  const [feedback, setFeedback] = useState("");

  // State for text-to-speech
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const audioRef = useRef(null);

  // Initialize speech recognition
  useEffect(() => {
    if ('webkitSpeechRecognition' in window) {
      const recognitionInstance = new window.webkitSpeechRecognition();
      recognitionInstance.continuous = true;
      recognitionInstance.interimResults = true;
      
      recognitionInstance.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }
        
        setUserSpeech(finalTranscript || interimTranscript);
      };
      
      recognitionInstance.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };
      
      recognitionInstance.onend = () => {
        if (isListening) {
          // If still in listening mode but recognition ended, restart it
          recognitionInstance.start();
        }
      };
      
      setRecognition(recognitionInstance);
    } else {
      alert("Your browser doesn't support speech recognition. Please use Chrome.");
    }
    
    return () => {
      if (recognition) {
        recognition.stop();
      }
    };
  }, [isListening]);

  // Initialize Gemini chat with system prompt
  useEffect(() => {
    const initChat = async () => {
      try {
        const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });
        
        // Create system prompt for JSON formatted questions
        const systemPrompt = `You are a technical interviewer conducting an interview for the position of ${jobTitle}. 
        The candidate should have the following skills: ${skills}.
        
        Your task:
        1. Generate exactly 7 interview questions covering:
           - Technical knowledge related to the required skills
           - Past projects and experience relevant to ${jobTitle}
           - Problem-solving abilities and technical approach
           - Career goals and motivation
           - Cultural fit and soft skills
           - Specific questions about items listed in skills: ${skills}
        
        2. VERY IMPORTANT: Return your questions in a valid JSON format as follows:
        {
          "introduction": "Brief introduction to the interview process",
          "questions": [
            {
              "id": 1,
              "question": "First question text here?",
              "category": "technical knowledge"
            },
            {
              "id": 2,
              "question": "Second question text here?",
              "category": "experience"
            },
            ... and so on for all 7 questions
          ],
          "conclusion": "Brief text to be shown when the interview is complete"
        }

        3. Guidelines:
           - Begin with easier questions and gradually increase difficulty
           - Make questions related to the job title and skills
           - Be professional, encouraging, and respectful throughout
           - Ensure the JSON is properly formatted with no syntax errors
           - Do not include any text outside of the JSON structure`;

        // Start chat with history configuration
        const chatInstance = model.startChat({
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
        setChat(chatInstance);
        
        // Send initial system prompt
        const result = await chatInstance.sendMessage(systemPrompt);
        const response = await result.response;
        
        try {
          // Parse the JSON response
          const jsonText = response.text().trim();
          const jsonData = JSON.parse(jsonText);
          
          setQuestions(jsonData.questions);
          setFeedback(jsonData.introduction);
          setIsLoading(false);
          
          // Generate audio for the first question after loading
          if (jsonData.questions && jsonData.questions.length > 0) {
            generateSpeech(jsonData.questions[0].question);
          }
        } catch (jsonError) {
          console.error('Error parsing JSON response:', jsonError);
          // Fallback: try to extract JSON from the response
          const jsonRegex = /{[\s\S]*}/;
          const match = response.text().match(jsonRegex);
          
          if (match) {
            try {
              const extractedJson = JSON.parse(match[0]);
              setQuestions(extractedJson.questions);
              setFeedback(extractedJson.introduction);
              setIsLoading(false);
              
              // Generate audio for the first question after loading
              if (extractedJson.questions && extractedJson.questions.length > 0) {
                generateSpeech(extractedJson.questions[0].question);
              }
            } catch (e) {
              throw new Error('Failed to parse JSON from response');
            }
          } else {
            throw new Error('No valid JSON found in response');
          }
        }
      } catch (error) {
        console.error('Error initializing chat:', error);
        setFeedback('There was an error connecting to the interview system. Please refresh and try again.');
        setIsLoading(false);
      }
    };
    
    initChat();
  }, [jobTitle, skills]);

  // Handle camera toggle
  const handleCameraToggle = async () => {
    if (!cameraOn) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setLocalStream(stream);
        setCameraOn(true);
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    } else {
      if (localStream) {
        localStream.getTracks().forEach((track) => track.stop());
      }
      setLocalStream(null);
      setCameraOn(false);
    }
  };

  // Handle microphone toggle for speech recognition
  const handleMicToggle = () => {
    if (!isListening) {
      setUserSpeech("");
      recognition.start();
      setIsListening(true);
    } else {
      recognition.stop();
      setIsListening(false);
    }
  };

  // Generate speech using ElevenLabs API
  const generateSpeech = async (text) => {
    try {
      const voice_id = "21m00Tcm4TlvDq8ikWAM"; // Default ElevenLabs voice ID
      const model_id = "eleven_monolingual_v1";
      
      const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voice_id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'xi-api-key': "sk_29a9c0c4e09ebff0cfb01b0cfb1bbbcdaa022d9505fb7e7d", // Use environment variable or replace with your key
        },
        body: JSON.stringify({
          text: text,
          model_id: model_id,
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.5
          }
        })
      });
      
      if (!response.ok) {
        throw new Error(`ElevenLabs API error: ${response.status}`);
      }
      
      const audioBlob = await response.blob();
      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);
      
      if (audioRef.current) {
        audioRef.current.src = url;
        audioRef.current.load();
      }
    } catch (error) {
      console.error('Error generating speech:', error);
    }
  };

  // Play/pause the audio
  const toggleAudio = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  // Listen for audio events
  useEffect(() => {
    const audio = audioRef.current;
    
    if (audio) {
      const handleEnded = () => setIsPlaying(false);
      audio.addEventListener('ended', handleEnded);
      
      return () => {
        audio.removeEventListener('ended', handleEnded);
      };
    }
  }, []);

  // Save the current answer
  const saveAnswer = () => {
    if (!userSpeech.trim()) return;
    
    const currentQuestion = questions[currentQuestionIndex];
    setAnswers(prev => ({
      ...prev,
      [currentQuestion.id]: userSpeech
    }));
    
    setUserSpeech("");
  };

  // Handle moving to the next question
  const handleNextQuestion = () => {
    // Save the current answer first
    saveAnswer();
    
    // Stop current audio if playing
    if (audioRef.current && isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    }
    
    // Move to the next question if not at the end
    if (currentQuestionIndex < questions.length - 1) {
      const nextIndex = currentQuestionIndex + 1;
      setCurrentQuestionIndex(nextIndex);
      
      // Generate audio for the next question
      generateSpeech(questions[nextIndex].question);
    } else {
      // Finish the interview
      finishInterview();
    }
  };

  // Finish the interview and generate feedback
  const finishInterview = async () => {
    setIsLoading(true);
    try {
      // Format answers for submission
      const answersFormatted = Object.entries(answers).map(([id, answer]) => {
        const question = questions.find(q => q.id.toString() === id);
        return {
          questionId: id,
          question: question ? question.question : "Unknown question",
          answer: answer
        };
      });
      
      // Send answers to Gemini for evaluation
      const promptForFeedback = `
      I've completed an interview for the ${jobTitle} position. 
      Here are my answers to your questions:
      
      ${JSON.stringify(answersFormatted, null, 2)}
      
      Please provide a concluding summary of the interview and any feedback.`;
      
      const result = await chat.sendMessage(promptForFeedback);
      const response = await result.response;
      
      const feedbackText = response.text();
      setFeedback(feedbackText);
      setInterviewComplete(true);
      
      // Generate audio for the feedback
      generateSpeech(feedbackText.substring(0, 1000)); // Limit to first 1000 chars if feedback is long
    } catch (error) {
      console.error('Error generating feedback:', error);
      setFeedback('Error generating feedback. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle manual update of speech text input
  const handleSpeechInputChange = (e) => {
    setUserSpeech(e.target.value);
  };

  // End the interview/call
  const endInterview = () => {
    // Stop all media
    if (localStream) {
      localStream.getTracks().forEach((track) => track.stop());
    }
    
    if (recognition) {
      recognition.stop();
    }
    
    // Stop audio if playing
    if (audioRef.current) {
      audioRef.current.pause();
    }
    
    // Reset states
    setLocalStream(null);
    setCameraOn(false);
    setIsListening(false);
    setIsPlaying(false);
    
    // Redirect or show end screen
    if (window.confirm("Are you sure you want to end the interview?")) {
      // Redirect to results or home page
      window.location.href = "/";
    }
  };

  return (
    <div style={styles.pageContainer}>
      {/* Hidden audio element for text-to-speech */}
      <audio ref={audioRef} />
      
      {/* Top Bar */}
      <div style={styles.topBar}>
        <div style={styles.meetingInfo}>
          <span style={styles.meetingTitle}>
            Interview for {jobTitle}
          </span>
        </div>
        <div style={styles.topBarRight}>
          <div style={styles.skillsTag}>
            Skills: {skills}
          </div>
        </div>
      </div>

      {/* Main Content: Two equal boxes side by side */}
      <div style={styles.mainContent}>
        {/* Camera Feed / Candidate Video */}
        <div style={styles.box}>
          {cameraOn ? (
            <video ref={videoRef} style={styles.video} autoPlay />
          ) : (
            <div style={styles.cameraPlaceholder}>
              <span style={styles.placeholderText}>Camera Off</span>
              {userSpeech && (
                <div style={styles.speechBubble}>
                  {userSpeech}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Interview Questions Box */}
        <div style={styles.aiBox}>
          {isLoading ? (
            <div style={styles.loadingContainer}>
              <div style={styles.loader}></div>
              <span>Loading interview questions...</span>
            </div>
          ) : interviewComplete ? (
            <div style={styles.feedbackContainer}>
              <h2 style={styles.feedbackTitle}>Interview Complete</h2>
              <div style={styles.feedbackContent}>
                <ReactMarkdown>{feedback}</ReactMarkdown>
              </div>
              <div style={styles.audioControls}>
                <button 
                  style={styles.audioButton}
                  onClick={toggleAudio}
                >
                  {isPlaying ? <VolumeX size={18} /> : <Volume2 size={18} />}
                  <span>{isPlaying ? "Stop Audio" : "Play Feedback"}</span>
                </button>
              </div>
              <button
                style={styles.endInterviewButton}
                onClick={endInterview}
              >
                Exit Interview
              </button>
            </div>
          ) : (
            <div style={styles.questionContainer}>
              <div style={styles.progressIndicator}>
                Question {currentQuestionIndex + 1} of {questions.length}
              </div>
              
              {questions.length > 0 && currentQuestionIndex < questions.length ? (
                <>
                  <div style={styles.questionBox}>
                    <div style={styles.questionHeader}>
                      <h3 style={styles.questionCategory}>
                        {questions[currentQuestionIndex].category}
                      </h3>
                      <button 
                        style={styles.audioButton}
                        onClick={toggleAudio}
                        title={isPlaying ? "Stop reading" : "Read question aloud"}
                      >
                        {isPlaying ? <VolumeX size={18} /> : <Volume2 size={18} />}
                      </button>
                    </div>
                    <p style={styles.questionText}>
                      {questions[currentQuestionIndex].question}
                    </p>
                  </div>
                  
                  {currentQuestionIndex === 0 && (
                    <div style={styles.introBox}>
                      <ReactMarkdown>{feedback}</ReactMarkdown>
                    </div>
                  )}
                  
                  <div style={styles.answerStatus}>
                    {answers[questions[currentQuestionIndex].id] ? (
                      <span style={styles.answerSaved}>Answer saved ✓</span>
                    ) : (
                      <span style={styles.answerPending}>Answer pending...</span>
                    )}
                  </div>
                </>
              ) : (
                <div style={styles.noQuestions}>
                  No questions available. Please refresh the page.
                </div>
              )}
              
              <div style={styles.navigationButtons}>
                <button
                  style={styles.nextButton}
                  onClick={handleNextQuestion}
                  disabled={!userSpeech.trim() && !answers[questions[currentQuestionIndex]?.id]}
                >
                  {currentQuestionIndex < questions.length - 1 ? (
                    <>Next Question <ArrowRight size={16} /></>
                  ) : (
                    <>Finish Interview</>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Speech Input Area */}
      <div style={styles.speechInputContainer}>
        <div style={styles.speechInputWrapper}>
          <input
            type="text"
            value={userSpeech}
            onChange={handleSpeechInputChange}
            placeholder={isListening ? "Listening..." : "Type or speak your response..."}
            style={styles.speechInput}
            disabled={isLoading || interviewComplete}
          />
          <button 
            style={styles.sendButton}
            onClick={saveAnswer}
            disabled={!userSpeech.trim() || isLoading || interviewComplete}
          >
            <Send size={18} />
          </button>
        </div>
      </div>

      {/* Bottom Bar */}
      <div style={styles.bottomBar}>
        <div style={styles.controls}>
          <div 
            style={{
              ...styles.controlIcon,
              backgroundColor: isListening ? '#ea4335' : '#5f6368',
              opacity: interviewComplete ? 0.5 : 1
            }}
            onClick={!interviewComplete ? handleMicToggle : undefined}
          >
            {isListening ? <MicOff size={18} /> : <Mic size={18} />}
            <span style={styles.controlLabel}>
              {isListening ? 'Stop' : 'Speak'}
            </span>
          </div>
          
          <div 
            style={{
              ...styles.controlIcon,
              backgroundColor: cameraOn ? '#5f6368' : '#5f6368'
            }}
            onClick={handleCameraToggle}
          >
            {cameraOn ? <VideoOff size={18} /> : <Video size={18} />}
            <span style={styles.controlLabel}>Camera</span>
          </div>
          
          <div 
            style={{
              ...styles.controlIcon,
              backgroundColor: '#ea4335'
            }}
            onClick={endInterview}
          >
            <PhoneOff size={18} />
            <span style={styles.controlLabel}>End</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const styles = {
  pageContainer: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    backgroundColor: "#202124",
    color: "#fff",
    fontFamily: "Arial, sans-serif",
  },
  topBar: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "0 16px",
    height: "50px",
    backgroundColor: "#3c4043",
  },
  meetingInfo: {
    display: "flex",
    alignItems: "center",
  },
  meetingTitle: {
    fontWeight: "bold",
    marginRight: "16px",
  },
  topBarRight: {
    display: "flex",
    alignItems: "center",
  },
  skillsTag: {
    marginLeft: "12px",
    padding: "4px 8px",
    backgroundColor: "#5f6368",
    borderRadius: "4px",
    fontSize: "0.8rem",
  },
  mainContent: {
    flex: 1,
    display: "flex",
    justifyContent: "space-around",
    alignItems: "center",
    backgroundColor: "#111",
    padding: "20px",
  },
  box: {
    width: "45%",
    height: "70%",
    backgroundColor: "#333",
    borderRadius: "8px",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    overflow: "hidden",
    position: "relative",
  },
  aiBox: {
    width: "45%",
    height: "70%",
    backgroundColor: "#333",
    borderRadius: "8px",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    position: "relative",
    padding: "20px",
  },
  video: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },
  cameraPlaceholder: {
    width: "100%",
    height: "100%",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    position: "relative",
  },
  placeholderText: {
    fontSize: "1.5rem",
    fontWeight: "bold",
    marginBottom: "20px",
  },
  speechBubble: {
    padding: "15px",
    backgroundColor: "rgba(255, 255, 255, 0.1)",
    borderRadius: "8px",
    margin: "10px",
    maxWidth: "80%",
    wordWrap: "break-word",
  },
  questionContainer: {
    display: "flex",
    flexDirection: "column",
    height: "100%",
    justifyContent: "space-between",
  },
  progressIndicator: {
    textAlign: "center",
    marginBottom: "15px",
    padding: "5px",
    backgroundColor: "#4285f4",
    borderRadius: "4px",
    fontSize: "0.9rem",
  },
  questionBox: {
    backgroundColor: "#4285f4",
    borderRadius: "8px",
    padding: "15px",
    marginBottom: "15px",
  },
  questionHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "10px",
  },
  questionCategory: {
    textTransform: "uppercase",
    fontSize: "0.8rem",
    opacity: "0.8",
    margin: 0,
  },
  questionText: {
    fontSize: "1.1rem",
    fontWeight: "bold",
  },
  introBox: {
    backgroundColor: "rgba(255, 255, 255, 0.1)",
    borderRadius: "8px",
    padding: "15px",
    marginBottom: "15px",
    maxHeight: "150px",
    overflowY: "auto",
  },
  answerStatus: {
    textAlign: "center",
    margin: "10px 0",
    fontSize: "0.9rem",
  },
  answerSaved: {
    color: "#8BC34A",
  },
  answerPending: {
    color: "#FFC107",
  },
  navigationButtons: {
    display: "flex",
    justifyContent: "flex-end",
    marginTop: "15px",
  },
  nextButton: {
    backgroundColor: "#4285f4",
    border: "none",
    color: "white",
    padding: "10px 15px",
    borderRadius: "4px",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    fontWeight: "bold",
  },
  feedbackContainer: {
    display: "flex",
    flexDirection: "column",
    height: "100%",
  },
  feedbackTitle: {
    textAlign: "center",
    marginBottom: "20px",
    color: "#4285f4",
  },
  feedbackContent: {
    backgroundColor: "rgba(255, 255, 255, 0.1)",
    borderRadius: "8px",
    padding: "15px",
    overflowY: "auto",
    flex: 1,
  },
  audioControls: {
    display: "flex",
    justifyContent: "center",
    margin: "15px 0",
  },
  audioButton: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "8px",
    backgroundColor: "#4285f4",
    color: "white",
    border: "none",
    borderRadius: "4px",
    padding: "8px 12px",
    cursor: "pointer",
    fontSize: "0.9rem",
  },
  endInterviewButton: {
    backgroundColor: "#ea4335",
    border: "none",
    color: "white",
    padding: "10px 15px",
    borderRadius: "4px",
    cursor: "pointer",
    marginTop: "20px",
    alignSelf: "center",
    fontWeight: "bold",
  },
  loadingContainer: {
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    height: "100%",
  },
  loader: {
    border: "4px solid rgba(255, 255, 255, 0.3)",
    borderRadius: "50%",
    borderTop: "4px solid #4285f4",
    width: "30px",
    height: "30px",
    animation: "spin 1s linear infinite",
    marginBottom: "10px",
  },
  "@keyframes spin": {
    "0%": { transform: "rotate(0deg)" },
    "100%": { transform: "rotate(360deg)" },
  },
  noQuestions: {
    textAlign: "center",
    marginTop: "20px",
    color: "#ea4335",
  },
  bottomBar: {
    height: "60px",
    backgroundColor: "#3c4043",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
  controls: {
    display: "flex",
    gap: "16px",
  },
  controlIcon: {
    padding: "8px 12px",
    backgroundColor: "#5f6368",
    borderRadius: "4px",
    cursor: "pointer",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    width: "60px",
  },
  controlLabel: {
    fontSize: "0.7rem",
    marginTop: "4px",
  },
  speechInputContainer: {
    padding: "10px 20px",
    backgroundColor: "#3c4043",
  },
  speechInputWrapper: {
    display: "flex",
    width: "100%",
    position: "relative",
  },
  speechInput: {
    flex: 1,
    padding: "10px 15px",
    backgroundColor: "#5f6368",
    color: "white",
    border: "none",
    borderRadius: "4px",
    fontSize: "1rem",
    outline: "none",
  },
  sendButton: {
    position: "absolute",
    right: "5px",
    top: "50%",
    transform: "translateY(-50%)",
    background: "none",
    border: "none",
    color: "white",
    cursor: "pointer",
    padding: "5px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
};

export default InterviewPage;