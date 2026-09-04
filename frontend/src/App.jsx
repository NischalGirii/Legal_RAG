import { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import './App.css';

// ---- SVG Icons ----
function SealMark() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2L3 7v6c0 5.5 3.8 10.7 9 12 5.2-1.3 9-6.5 9-12V7l-9-5z" />
      <path d="M12 8v4" />
      <path d="M12 16h.01" />
    </svg>
  );
}

function ThemeIcon({ dark }) {
  return dark ? (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="5" />
      <line x1="12" y1="1" x2="12" y2="3" />
      <line x1="12" y1="21" x2="12" y2="23" />
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
      <line x1="1" y1="12" x2="3" y2="12" />
      <line x1="21" y1="12" x2="23" y2="12" />
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
    </svg>
  ) : (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function MicIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}

function MicIconLarge() {
  return (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}

function VolumeIcon({ on }) {
  return on ? (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07" />
    </svg>
  ) : (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <line x1="23" y1="9" x2="17" y2="15" />
      <line x1="17" y1="9" x2="23" y2="15" />
    </svg>
  );
}

function ReplayIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
      <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
    </svg>
  );
}

function EndCallIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10.68 13.31a16 16 0 0 0 3.41 2.6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7 2 2 0 0 1 1.72 2v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.42 19.42 0 0 1-3.33-2.67m-2.67-3.34a19.79 19.79 0 0 1-3.07-8.63A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91" />
      <line x1="23" y1="1" x2="1" y2="23" />
    </svg>
  );
}

// ---- Turn VAD Parameters ----
const SILENCE_RMS_THRESHOLD = 0.02;
const SILENCE_HOLD_MS = 1200;
const MIN_TURN_MS = 500;
const MAX_TURN_MS = 18000;

// Truncate helper to keep the call screen compact
function truncateText(text, maxChars = 110) {
  if (!text) return '';
  const clean = text.replace(/[*#_`]/g, '').trim();
  if (clean.length <= maxChars) return clean;
  return clean.substring(0, maxChars).trim() + '...';
}

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'नमस्कार! म तपाईंको नेपाली कानूनी सहायक हुँ। कुनै प्रश्न सोध्नुहोस्।' },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [voiceInputSupported, setVoiceInputSupported] = useState(true);
  const [autoSpeak, setAutoSpeak] = useState(false);

  const [currentCase, setCurrentCase] = useState(null);
  const [sessionId] = useState(() => {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
    return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
  });

  const [callActive, setCallActive] = useState(false);
  const [callState, setCallState] = useState('idle'); // 'idle' | 'listening' | 'thinking' | 'speaking'
  const [callError, setCallError] = useState('');

  // Audio & Hardware Refs
  const audioContextRef = useRef(null);
  const micStreamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const animFrameRef = useRef(null);
  const isRecordingTurnRef = useRef(false);
  const silenceStartRef = useRef(null);
  const turnStartRef = useRef(null);
  const callActiveRef = useRef(false);
  const callStateRef = useRef('idle');
  const currentAudioRef = useRef(null);
  const endRef = useRef(null);

  useEffect(() => {
    callStateRef.current = callState;
  }, [callState]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setVoiceInputSupported(false);
    }
  }, []);

  // ---------- Native Neural Nepali Text-to-Speech ----------
  const speak = useCallback((text) => {
    return new Promise((resolve) => {
      if (!text || !text.trim()) {
        resolve();
        return;
      }

      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current = null;
      }

      try {
        const url = `http://localhost:8000/api/tts?text=${encodeURIComponent(text)}`;
        const audio = new Audio(url);
        currentAudioRef.current = audio;

        audio.onended = () => {
          currentAudioRef.current = null;
          resolve();
        };
        audio.onerror = (err) => {
          console.error('TTS playback error:', err);
          currentAudioRef.current = null;
          resolve();
        };

        audio.play().catch((err) => {
          console.error('TTS play call failed:', err);
          currentAudioRef.current = null;
          resolve();
        });
      } catch (err) {
        console.error('TTS request failed:', err);
        currentAudioRef.current = null;
        resolve();
      }
    });
  }, []);

  useEffect(() => {
    if (!autoSpeak || loading || callActiveRef.current) return;
    const last = messages[messages.length - 1];
    if (last?.role === 'assistant') speak(last.content);
  }, [messages, loading, autoSpeak, speak]);

  const toggleAutoSpeak = () => {
    if (autoSpeak && currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    setAutoSpeak((v) => !v);
  };

  // ---------- Case Tracking & RAG ----------
  const extractDecisionNumber = (text) => {
    const match = text.match(/निर्णय\s*नं\.?\s*(\d+)/);
    if (match) return match[1];
    if (text.match(/निर्णय|मुद्दा|फैसला/)) {
      const numMatch = text.match(/\b(\d{3,4})\b/);
      if (numMatch) return numMatch[1];
    }
    return null;
  };

  const updateCaseFromText = (text) => {
    const dec = extractDecisionNumber(text);
    if (dec) setCurrentCase({ case_id: `decision_${dec}`, decision_no: dec });
  };

  const fetchReply = async (text) => {
    const payload = {
      message: text,
      session_id: sessionId,
      case_id: currentCase?.case_id || null,
      decision_no: currentCase?.decision_no || null,
    };
    const res = await axios.post('http://localhost:8000/api/chat', payload);
    return res.data.reply;
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    const userText = input.trim();
    updateCaseFromText(userText);
    setMessages((prev) => [...prev, { role: 'user', content: userText }]);
    setInput('');
    setLoading(true);
    try {
      const reply = await fetchReply(userText);
      updateCaseFromText(reply);
      setMessages((prev) => [...prev, { role: 'assistant', content: reply }]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [...prev, { role: 'assistant', content: 'सर्भर त्रुटि भयो। कृपया पुन: प्रयास गर्नुहोस्।' }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // ---------- Clean Up On End Call ----------
  const endCall = useCallback(() => {
    callActiveRef.current = false;

    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try { mediaRecorderRef.current.stop(); } catch (e) {}
      mediaRecorderRef.current = null;
    }
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((t) => t.stop());
      micStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }

    setCallActive(false);
    setCallState('idle');
    setCallError('');
  }, []);

  // ---------- Continuous Multi-Turn VAD Audio Engine ----------
  const calculateRMS = (buffer) => {
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
    return Math.sqrt(sum / buffer.length);
  };

  const processAudioTurn = async (audioBlob, resumeListeningFn) => {
    if (!callActiveRef.current) return;
    setCallState('thinking');

    try {
      // 1. Transcribe
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      const trRes = await axios.post('http://localhost:8000/api/transcribe', formData);
      const userText = trRes.data.transcript?.trim();

      if (!userText || !callActiveRef.current) {
        if (callActiveRef.current && resumeListeningFn) resumeListeningFn();
        return;
      }

      setMessages((prev) => [...prev, { role: 'user', content: userText }]);
      updateCaseFromText(userText);

      // 2. Query Hybrid Legal RAG
      const botReply = await fetchReply(userText);
      if (!callActiveRef.current) return;

      updateCaseFromText(botReply);
      setMessages((prev) => [...prev, { role: 'assistant', content: botReply }]);

      // 3. Play Speech
      setCallState('speaking');
      await speak(botReply);

      // 4. Multi-Turn: Automatically resume listening for the next question
      if (callActiveRef.current) {
        setTimeout(() => {
          if (callActiveRef.current && resumeListeningFn) {
            resumeListeningFn();
          }
        }, 400);
      }
    } catch (err) {
      console.error('Turn error:', err);
      setCallError('प्रक्रियामा त्रुटि भयो। पुन: सोध्नुहोस्।');
      if (callActiveRef.current && resumeListeningFn) {
        resumeListeningFn();
      }
    }
  };

  const startContinuousCall = async () => {
    if (!voiceInputSupported || callActive) return;
    setCallError('');
    callActiveRef.current = true;
    setCallActive(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      micStreamRef.current = stream;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx();
      audioContextRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);

      const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4';
      const buffer = new Float32Array(analyser.fftSize);

      const startNewTurnRecording = () => {
        if (!callActiveRef.current) return;

        if (ctx.state === 'suspended') {
          ctx.resume();
        }

        audioChunksRef.current = [];
        isRecordingTurnRef.current = false;
        silenceStartRef.current = null;
        turnStartRef.current = null;

        const recorder = new MediaRecorder(stream, { mimeType });
        mediaRecorderRef.current = recorder;

        recorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) {
            audioChunksRef.current.push(e.data);
          }
        };

        recorder.onstop = () => {
          const chunks = [...audioChunksRef.current];
          audioChunksRef.current = [];
          if (callActiveRef.current && chunks.length > 0) {
            const blob = new Blob(chunks, { type: mimeType });
            if (blob.size > 1500) {
              processAudioTurn(blob, startNewTurnRecording);
              return;
            }
          }
          if (callActiveRef.current) {
            startNewTurnRecording();
          }
        };

        try {
          recorder.start(100);
        } catch (e) {
          console.error(e);
        }
        setCallState('listening');
      };

      const vadLoop = () => {
        if (!callActiveRef.current) return;

        analyser.getFloatTimeDomainData(buffer);
        const rms = calculateRMS(buffer);
        const now = Date.now();

        if (callStateRef.current === 'listening') {
          if (rms > SILENCE_RMS_THRESHOLD) {
            silenceStartRef.current = null;
            if (!isRecordingTurnRef.current) {
              isRecordingTurnRef.current = true;
              turnStartRef.current = now;
            }
          } else if (isRecordingTurnRef.current) {
            if (!silenceStartRef.current) silenceStartRef.current = now;
            const silenceDuration = now - silenceStartRef.current;
            const turnDuration = now - turnStartRef.current;

            if (silenceDuration > SILENCE_HOLD_MS || turnDuration > MAX_TURN_MS) {
              isRecordingTurnRef.current = false;
              silenceStartRef.current = null;

              if (turnDuration >= MIN_TURN_MS && mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
                try { mediaRecorderRef.current.stop(); } catch (e) {}
              }
            }
          }
        }

        animFrameRef.current = requestAnimationFrame(vadLoop);
      };

      startNewTurnRecording();
      vadLoop();
    } catch (err) {
      console.error('Call initialization failed:', err);
      setCallError('माइक्रोफोन पहुँच अस्वीकृत भयो।');
      endCall();
    }
  };

  useEffect(() => {
    return () => {
      if (callActiveRef.current) endCall();
    };
  }, [endCall]);

  const callStatusText = {
    listening: 'सुन्दै...',
    thinking: 'प्रमाण खोज्दै...',
    speaking: 'जवाफ दिँदै...',
    idle: 'तयार',
  }[callState];

  const lastUserMsg = [...messages].reverse().find((m) => m.role === 'user')?.content || '';
  const lastAssistantMsg = [...messages].reverse().find((m) => m.role === 'assistant')?.content || '';

  return (
    <div className="app" data-theme={darkMode ? 'dark' : 'light'}>
      <header className="letterhead">
        <div className="brand">
          <span className="seal"><SealMark /></span>
          <div className="brand-text">
            <h1>औपचारिक सहायक</h1>
            <p>नेपाली कानून सहायता · जानकारी मात्र, सल्लाह होइन</p>
          </div>
        </div>
        <div className="toggle-group">
          <button
            type="button"
            className={`icon-toggle ${autoSpeak ? 'icon-toggle--active' : ''}`}
            onClick={toggleAutoSpeak}
            aria-pressed={autoSpeak}
            aria-label={autoSpeak ? 'स्वतः पढ्ने बन्द गर्नुहोस्' : 'जवाफ स्वतः पढ्नुहोस्'}
            title={autoSpeak ? 'स्वतः पढ्ने: चालु' : 'स्वतः पढ्ने: बन्द'}
          >
            <VolumeIcon on={autoSpeak} />
          </button>
          <button
            type="button"
            className="icon-toggle"
            onClick={() => setDarkMode((d) => !d)}
            aria-label={darkMode ? 'उज्यालो मोडमा जानुहोस्' : 'अँध्यारो मोडमा जानुहोस्'}
          >
            <ThemeIcon dark={darkMode} />
          </button>
        </div>
      </header>

      <div className="transcript">
        {messages.map((msg, idx) => (
          <div key={idx} className={`entry ${msg.role === 'user' ? 'entry--user' : 'entry--assistant'}`}>
            <div className="entry-label">{msg.role === 'user' ? 'तपाईं' : 'सहायक'}</div>
            <div className="entry-content">
              {msg.content}
              {msg.role === 'assistant' && (
                <button
                  type="button"
                  className="replay-button"
                  onClick={() => speak(msg.content)}
                  aria-label="फेरि सुन्नुहोस्"
                  title="फेरि सुन्नुहोस्"
                >
                  <ReplayIcon />
                </button>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="entry entry--assistant entry--loading">
            <div className="entry-label">सहायक</div>
            <div className="entry-content">
              <span className="dot" /><span className="dot" /><span className="dot" />
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="composer">
        {callError && <p className="dictate-status">{callError}</p>}
        <div className="composer-row">
          <textarea
            className="composer-field"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="तपाईंको कानूनी प्रश्न लेख्नुहोस्..."
            rows={1}
            disabled={loading}
          />
          <button
            type="button"
            className="mic-button"
            onClick={startContinuousCall}
            disabled={loading || !voiceInputSupported || callActive}
            aria-label="फोन कल सुरु गर्नुहोस्"
            title={voiceInputSupported ? 'निरन्तर भ्वाइस कल सुरु गर्नुहोस्' : 'माइक्रोफोन उपलब्ध छैन'}
          >
            <MicIcon />
          </button>
          <button
            type="button"
            className="composer-send"
            onClick={sendMessage}
            disabled={loading || !input.trim()}
          >
            पठाउनुहोस् <SendIcon />
          </button>
        </div>
        <p className="disclaimer">यहाँ दिइएको जानकारी कानूनी सल्लाहको विकल्प होइन।</p>
      </div>

      {/* Structured Call Overlay with Clean Dialogue Card */}
      {callActive && (
        <div className="call-overlay" role="dialog" aria-modal="true" aria-label="भ्वाइस कल">
          <div className="call-panel">
            <span className="call-seal"><SealMark /></span>
            <p className="call-brand">औपचारिक सहायक</p>
            
            <div className={`call-orb call-orb--${callState}`}>
              <MicIconLarge />
            </div>
            
            <p className="call-status">
              {callStatusText}
              {callState === 'thinking' && <span className="call-dots">...</span>}
            </p>

            {/* Compact Live Dialogue Card */}
            <div className="call-dialogue-card">
              {lastUserMsg && (
                <div className="call-dialogue-row call-dialogue-row--user">
                  <span className="call-badge">तपाईं:</span>
                  <span className="call-dialogue-text">{truncateText(lastUserMsg, 75)}</span>
                </div>
              )}
              <div className="call-dialogue-row call-dialogue-row--assistant">
                <span className="call-badge">सहायक:</span>
                <span className="call-dialogue-text">
                  {callState === 'thinking' && 'उत्तर खोज्दैछु...'}
                  {callState === 'listening' && (!lastAssistantMsg ? 'प्रश्न सोध्नुहोस्...' : truncateText(lastAssistantMsg, 110))}
                  {callState === 'speaking' && truncateText(lastAssistantMsg, 110)}
                </span>
              </div>
            </div>

            <button type="button" className="call-end-button" onClick={endCall} aria-label="कल अन्त्य गर्नुहोस्">
              <EndCallIcon />
            </button>
            <p className="call-end-label">कल अन्त्य गर्नुहोस्</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;