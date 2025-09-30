import React, { useState, useRef, useEffect } from 'react'
import MessageList from './MessageList'
import InputBox from './InputBox'
import { chatAPI } from '../services/api'

const ChatInterface = () => {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async (message) => {
    const userMessage = { 
      text: message, 
      sender: 'user',
      timestamp: new Date().toISOString()
    }
    
    const updatedMessages = [...messages, userMessage]
    setMessages(updatedMessages)
    setIsLoading(true)

    try {
      const response = await chatAPI.sendMessage(message)
      const botMessage = { 
        text: response.reply, 
        sender: 'bot',
        timestamp: new Date().toISOString()
      }
      setMessages([...updatedMessages, botMessage])
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage = { 
        text: 'Sorry, I encountered an error. Please try again.', 
        sender: 'bot',
        timestamp: new Date().toISOString(),
        isError: true
      }
      setMessages([...updatedMessages, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h1>🦙 Local Llama Chatbot</h1>
      </div>
      <MessageList messages={messages} isLoading={isLoading} />
      <div ref={messagesEndRef} />
      <InputBox onSendMessage={handleSendMessage} isLoading={isLoading} />
    </div>
  )
}

export default ChatInterface