import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './App.css'

const API_URL = 'http://localhost:8000/api'

function App() {
  const [sessionId] = useState(() => `session-${Date.now()}`)
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [totalInfluence, setTotalInfluence] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [gameWon, setGameWon] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading || gameWon) return

    const userMessage = inputValue.trim()
    setInputValue('')
    setIsLoading(true)
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])

    try {
      const { data } = await axios.post(`${API_URL}/chat`, { session_id: sessionId, message: userMessage })
      setTotalInfluence(data.total_influence)
      setMessages(prev => [...prev, { role: 'assistant', content: data.doorman_reply, influence_score: data.influence_score, reasoning: data.reasoning }])
      if (data.game_won) setGameWon(true)
    } catch (error) {
      console.error('Error sending message:', error)
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.', error: true }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const resetGame = async () => {
    try {
      await axios.post(`${API_URL}/session/${sessionId}/reset`)
      setMessages([])
      setTotalInfluence(0)
      setGameWon(false)
    } catch (error) {
      console.error('Error resetting game:', error)
    }
  }

  const getInfluenceColor = () => totalInfluence >= 100 ? '#10b981' : totalInfluence >= 50 ? '#3b82f6' : totalInfluence >= 0 ? '#f59e0b' : '#ef4444'
  const getInfluencePercentage = () => Math.min(Math.max(totalInfluence, 0), 100)

  return (
    <div className="app">
      <div className="game-container">
        <div className="header">
          <h1> The Doorman Game</h1>
          <p className="subtitle">Persuade Arthur, the doorman, to let you into the exclusive Dubai nightclub</p>
        </div>

        <div className="influence-meter-container">
          <div className="influence-label">
            <span>Influence Meter</span>
            <span className="influence-score">{totalInfluence}/100</span>
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
          {gameWon && (
            <div className="win-message">
              You've convinced the doorman! Access granted!
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
              onKeyPress={handleKeyPress}
              placeholder={gameWon ? "Game complete! Reset to play again." : "Type your message to the doorman..."}
              disabled={isLoading || gameWon}
              rows={2}
            />
            <div className="button-group">
              <button
                onClick={sendMessage}
                disabled={!inputValue.trim() || isLoading || gameWon}
                className="send-button"
              >
                {isLoading ? 'Sending...' : 'Send'}
              </button>
              <button onClick={resetGame} className="reset-button">
                Reset
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

