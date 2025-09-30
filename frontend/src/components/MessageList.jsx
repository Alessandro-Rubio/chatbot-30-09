import React from 'react'

const MessageList = ({ messages, isLoading }) => {
  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <div className="message-list">
      {messages.map((message, index) => (
        <div key={index} className={`message ${message.sender}`}>
          <div className="message-content">
            {message.text}
          </div>
          <div className="message-time">
            {formatTime(message.timestamp)}
          </div>
        </div>
      ))}
      {isLoading && (
        <div className="message bot loading">
          <div className="message-content">
            Thinking
            <span className="typing-indicator">
              <span className="typing-dot"></span>
              <span className="typing-dot"></span>
              <span className="typing-dot"></span>
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

export default MessageList