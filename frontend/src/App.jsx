import React, { useEffect, useRef, useState } from 'react'
import axios from 'axios'
import './App.css'

const API_BASE =
  (import.meta?.env?.VITE_API_URL) ||
  (typeof process !== 'undefined' ? process.env.REACT_APP_API_URL : '') ||
  'http://localhost:8000'

const api = axios.create({
  baseURL: `${API_BASE}/api`,
})

const makeSessionId = () => `session-${Date.now()}`

function App() {
  const [sessionId, setSessionId] = useState(() => {
    const saved = localStorage.getItem('doorman_session_id')
    const sid = saved || makeSessionId()
    localStorage.setItem('doorman_session_id', sid)
    return sid
  })

  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [totalInfluence, setTotalInfluence] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [gameWon, setGameWon] = useState(false)
  const [gameLost, setGameLost] = useState(false)
  const [apiError, setApiError] = useState('')
  const [meterMin, setMeterMin] = useState(-30)
  const [meterMax, setMeterMax] = useState(30)
  const [winThreshold, setWinThreshold] = useState(30)

  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isLoading])

  // Load config + restore session on mount / session change
  useEffect(() => {
    let cancelled = false

    const load = async () => {
      setApiError('')
      try {
        const root = await axios.get(`${API_BASE}/`)
        if (!cancelled && root?.data) {
          setMeterMin(root.data.meter_min ?? -30)
          setMeterMax(root.data.meter_max ?? 30)
          setWinThreshold(root.data.win_threshold ?? 30)
        }
      } catch {}

      try {
        const { data } = await api.get(`/session/${sessionId}`)
        if (cancelled) return

        setTotalInfluence(data.total_influence ?? 0)
        const conv = Array.isArray(data.conversation) ? data.conversation : []
        setMessages(
          conv.map(m => ({
            role: m.role,
            content: m.content,
          }))
        )

        const ti = data.total_influence ?? 0
        setGameWon(ti >= (winThreshold || 30))
        setGameLost(ti <= (meterMin || -30))
      } catch (e) {
        if (!cancelled) setApiError('Failed to restore session.')
      }
    }

    load()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId])

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading || gameWon || gameLost) return

    const userMessage = inputValue.trim()
    setInputValue('')
    setIsLoading(true)
    setApiError('')

    setMessages(prev => [...prev, { role: 'user', content: userMessage }])

    try {
      const { data } = await api.post(`/chat`, { session_id: sessionId, message: userMessage })

      setTotalInfluence(data.total_influence)
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: data.doorman_reply,
          influence_score: data.influence_score,
          reasoning: data.reasoning,
        },
      ])

      if (data.game_won) setGameWon(true)
      if (data.game_lost) setGameLost(true)
      if (data.error) setApiError(data.error)
    } catch (error) {
      const msg =
        error?.response?.data?.detail ||
        error?.message ||
        'API error. Please try again.'
      setApiError(msg)
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.', error: true },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const resetGame = async () => {
    try {
      await api.post(`/session/${sessionId}/reset`)
      setMessages([])
      setTotalInfluence(0)
      setGameWon(false)
      setGameLost(false)
      setApiError('')
    } catch (error) {
      setApiError('Error resetting game.')
    }
  }

  const newGame = async () => {
    const sid = makeSessionId()
    localStorage.setItem('doorman_session_id', sid)
    setSessionId(sid)
    try {
      await api.post(`/session/${sid}/reset`)
    } catch {}
    setMessages([])
    setTotalInfluence(0)
    setGameWon(false)
    setGameLost(false)
    setApiError('')
  }

  const getInfluenceColor = () =>
    totalInfluence >= winThreshold ? '#10b981'
      : totalInfluence >= 0 ? '#3b82f6'
      : totalInfluence >= (meterMin * 0.5) ? '#f59e0b'
      : '#ef4444'

  const getInfluencePercentage = () => {
    const min = meterMin
    const max = meterMax
    const clamped = Math.max(min, Math.min(max, totalInfluence))
    const denom = (max - min) || 1
    return ((clamped - min) / denom) * 100
  }

  return (
    <div className="app">
      <div className="game-container">
        <div className="header">
          <h1>The Doorman Game</h1>
          <p className="subtitle">Persuade Arthur to let you into the exclusive Dubai nightclub</p>
        </div>

        <div className="influence-meter-container">
          <div className="influence-label">
            <span>Influence Meter</span>
            <span className="influence-score">{totalInfluence}/{meterMax}</span>
          </div>

          <div className="influence-bar-wrapper">
            <div
              className="influence-bar"
              style={{
                width: `${getInfluencePercentage()}%`,
                backgroundColor: getInfluenceColor(),
              }}
            />
          </div>

          {apiError && (
            <div className="win-message" style={{ background: '#fff3cd', color: '#664d03' }}>
              {apiError}
            </div>
          )}

          {gameWon && (
            <div className="win-message">
              You’ve convinced the doorman! Access granted!
            </div>
          )}

          {gameLost && (
            <div className="win-message" style={{ background: '#ffe4e6', color: '#9f1239' }}>
              You pushed it too far. Arthur refuses entry. Reset or start a new game.
            </div>
          )}
        </div>

        <div className="chat-container">
          <div className="messages">
            {messages.length === 0 && (
              <div className="welcome-message">
                <p>Start the conversation!</p>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role}`}>
                <div className="message-content">
                  {msg.content}
                </div>
                {msg.influence_score !== undefined && (
                  <div className={`score-indicator ${msg.influence_score >= 0 ? 'positive' : 'negative'}`}>
                    {msg.influence_score > 0 ? '+' : ''}{msg.influence_score}
                  </div>
                )}
              </div>
            ))}

            {isLoading && (
              <div className="message assistant">
                <div className="message-content typing">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <div className="input-container">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={gameWon || gameLost ? "Game complete! Reset or New Game to play again." : "Type your message to the doorman..."}
              disabled={isLoading || gameWon || gameLost}
              rows={2}
            />

            <div className="button-group">
              <button
                onClick={sendMessage}
                disabled={!inputValue.trim() || isLoading || gameWon || gameLost}
                className="send-button"
              >
                {isLoading ? 'Sending...' : 'Send'}
              </button>

              <button onClick={resetGame} className="reset-button">
                Reset
              </button>

              <button onClick={newGame} className="reset-button">
                New Game
              </button>
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}

export default App
