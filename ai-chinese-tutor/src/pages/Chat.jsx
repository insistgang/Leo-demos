import { useState, useEffect, useRef } from 'react'
import { Card, Input, Button, List, Typography, Spin, message } from 'antd'
import { SendOutlined, LoadingOutlined, UserOutlined, RobotOutlined, ClearOutlined, CloseOutlined, CheckCircleOutlined } from '@ant-design/icons'

const { TextArea } = Input
const { Title, Text, Paragraph } = Typography

import { aiChatService } from '../services/aiChatService'
import { studyRecordService } from '../services/studyRecordService'

const Chat = () => {
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [chatHistory, setChatHistory] = useState([])
  const [currentChatId, setCurrentChatId] = useState(null)
  const messagesEndRef = useRef(null)
  const typewriterRef = useRef(null)

  // 从本地存储加载聊天历史
  useEffect(() => {
    loadChatHistory()
  }, [])

  // 自动滚动到最新消息
  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // 组件卸载时清除定时器
  useEffect(() => {
    return () => {
      if (typewriterRef.current) {
        clearInterval(typewriterRef.current)
      }
    }
  }, [])

  const loadChatHistory = () => {
    const saved = localStorage.getItem('chatHistory')
    if (saved) {
      try {
        const history = JSON.parse(saved)
        setChatHistory(history)
        // 如果有历史记录，加载最新的一次聊天
        if (history.length > 0) {
          const latestChat = history[history.length - 1]
          setCurrentChatId(latestChat.id)
          setMessages(latestChat.messages)
        }
      } catch (error) {
        console.error('加载聊天历史失败:', error)
        message.error('加载聊天历史失败')
      }
    }
  }



  const saveChatHistory = () => {
    if (messages.length === 0) return

    // 检查是否已存在相同ID的聊天
    const existingIndex = chatHistory.findIndex(chat => chat.id === currentChatId)
    const updatedChat = {
      id: currentChatId || Date.now(),
      title: messages[0].content.substring(0, 30) + (messages[0].content.length > 30 ? '...' : ''),
      messages: messages,
      date: new Date().toLocaleString('zh-CN')
    }

    let newHistory
    if (existingIndex >= 0) {
      newHistory = [...chatHistory]
      newHistory[existingIndex] = updatedChat
    } else {
      newHistory = [updatedChat, ...chatHistory.slice(0, 9)] // 只保留最近10条聊天记录
      setCurrentChatId(updatedChat.id)
    }

    setChatHistory(newHistory)
    localStorage.setItem('chatHistory', JSON.stringify(newHistory))
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  // 打字机效果函数
  const typewriterEffect = (fullText, messageId) => {
    let currentIndex = 0
    const delay = 30 // 每个字符的延迟时间（毫秒）

    // 清除之前的定时器
    if (typewriterRef.current) {
      clearInterval(typewriterRef.current)
    }

    // 创建定时器逐字显示文本
    typewriterRef.current = setInterval(() => {
      if (currentIndex <= fullText.length) {
        const partialText = fullText.substring(0, currentIndex)
        
        setMessages(prevMessages => 
          prevMessages.map(msg => 
            msg.id === messageId 
              ? { ...msg, content: partialText, isComplete: currentIndex === fullText.length } 
              : msg
          )
        )
        
        currentIndex++
      } else {
        // 文本显示完成，清除定时器
        clearInterval(typewriterRef.current)
        typewriterRef.current = null
        saveChatHistory()
      }
    }, delay)
  }

  // 发送消息
  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date().toLocaleTimeString()
    }

    setMessages([...messages, userMessage])
    setInputValue('')
    setIsLoading(true)

    try {
      // 创建一个加载中的机器人消息
      const loadingBotMessageId = Date.now() + 1
      const loadingBotMessage = {
        id: loadingBotMessageId,
        type: 'bot',
        content: '',
        timestamp: new Date().toLocaleTimeString(),
        isComplete: false
      }
      
      setMessages(prev => [...prev, loadingBotMessage])
      
      // 调用DeepSeek API
      const response = await callDeepSeekAPI(userMessage.content)
      
      // 开始打字机效果
      typewriterEffect(response, loadingBotMessageId)
    } catch (error) {
      // 添加错误消息
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: '很抱歉，获取响应失败，请稍后重试。',
        timestamp: new Date().toLocaleTimeString(),
        isComplete: true
      }
      setMessages(prev => [...prev, errorMessage])
      message.error('获取响应失败，请稍后重试')
      console.error('API调用失败:', error)
    } finally {
      setIsLoading(false)
    }
  }

  // 调用DeepSeek API
  const callDeepSeekAPI = async (prompt) => {
    try {
      // 构建消息
      const messages = [
        {
          role: 'system',
          content: '你是一个专业的语文学习顾问，擅长回答各种语文相关问题，并能提供个性化的学习建议。'
        },
        {
          role: 'user',
          content: prompt
        }
      ]
      
      // 调用API
      const response = await aiChatService.callDeepSeekAPI(messages)
      return response
    } catch (error) {
      console.error('DeepSeek API调用失败:', error)
      throw error
    }
  }

  // 讨论学习情况
  const discussStudyProgress = async () => {
    setIsLoading(true)
    
    try {
      // 获取学习数据
      const statistics = await studyRecordService.getStudyStatistics()
      const categoryData = await studyRecordService.getCategoryStudyData()
      const recentRecords = await studyRecordService.getStudyRecords({ pageSize: 5 })
      
      // 构建学习数据对象
      const studyData = {
        statistics,
        categoryData,
        recentRecords: recentRecords.data
      }
      
      // 创建用户消息
      const userMessage = {
        id: Date.now(),
        type: 'user',
        content: '请分析我的学习情况并提供建议',
        timestamp: new Date().toLocaleTimeString()
      }
      
      setMessages(prev => [...prev, userMessage])
      
      // 创建一个加载中的机器人消息
      const loadingBotMessageId = Date.now() + 1
      const loadingBotMessage = {
        id: loadingBotMessageId,
        type: 'bot',
        content: '',
        timestamp: new Date().toLocaleTimeString(),
        isComplete: false
      }
      
      setMessages(prev => [...prev, loadingBotMessage])
      
      // 调用AI服务分析学习情况
      const response = await aiChatService.discussStudyProgress(studyData)
      
      // 开始打字机效果
      typewriterEffect(response, loadingBotMessageId)
    } catch (error) {
      // 添加错误消息
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: '很抱歉，获取学习分析失败，请稍后重试。',
        timestamp: new Date().toLocaleTimeString(),
        isComplete: true
      }
      setMessages(prev => [...prev, errorMessage])
      message.error('获取学习分析失败，请稍后重试')
      console.error('学习分析失败:', error)
    } finally {
      setIsLoading(false)
    }
  }

  // 新建聊天
  const startNewChat = () => {
    setMessages([])
    setCurrentChatId(null)
  }

  // 加载历史聊天
  const loadChat = (chat) => {
    setCurrentChatId(chat.id)
    setMessages(chat.messages)
  }

  // 删除历史聊天
  const deleteChat = (chatId, e) => {
    e.stopPropagation()
    const newHistory = chatHistory.filter(chat => chat.id !== chatId)
    setChatHistory(newHistory)
    localStorage.setItem('chatHistory', JSON.stringify(newHistory))
    
    // 如果删除的是当前聊天，新建一个聊天
    if (chatId === currentChatId) {
      startNewChat()
    }
  }

  // 清空当前聊天
  const clearCurrentChat = () => {
    setMessages([])
  }



  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <RobotOutlined className="text-blue-600 mr-2" />
              <Title level={2}>智能语文助手</Title>
            </div>
            <div className="flex space-x-2">
              <Button onClick={discussStudyProgress} type="default" icon={<CheckCircleOutlined />}>学习分析</Button>
              <Button onClick={clearCurrentChat} icon={<ClearOutlined />}>清空聊天</Button>
              <Button type="primary" onClick={startNewChat}>新建聊天</Button>
            </div>
          </div>
        
        <div className="flex flex-col md:flex-row gap-6">
          {/* 聊天历史侧边栏 */}
          <Card className="md:w-1/4 h-[500px] overflow-y-auto">
            <Title level={5} className="mb-4">聊天历史</Title>
            {chatHistory.length === 0 ? (
              <Text type="secondary">暂无聊天记录</Text>
            ) : (
              <List
                dataSource={chatHistory}
                renderItem={(chat) => (
                  <List.Item
                    className={`cursor-pointer p-2 rounded mb-2 hover:bg-gray-100 ${currentChatId === chat.id ? 'bg-blue-50' : ''}`}
                    onClick={() => loadChat(chat)}
                  >
                    <div className="w-full relative">
                      <div className="font-medium line-clamp-1">{chat.title}</div>
                      <Text type="secondary" className="text-xs">{chat.date}</Text>
                      <Button
                        type="text"
                        size="small"
                        icon={<CloseOutlined />}
                        className="absolute right-0 top-0 text-gray-400 hover:text-red-500"
                        onClick={(e) => deleteChat(chat.id, e)}
                      />
                    </div>
                  </List.Item>
                )}
              />
            )}
          </Card>
          
          {/* 聊天主界面 */}
          <Card className="md:w-3/4 flex flex-col h-[500px]">
            {/* 聊天内容区域 */}
            <div className="flex-1 overflow-y-auto p-4">
              {messages.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-gray-400">
                  <RobotOutlined className="text-4xl mb-4" />
                  <Text>你好！我是你的智能语文助手，有什么问题可以问我哦~</Text>
                </div>
              ) : (
                <List
                  dataSource={messages}
                  renderItem={(message) => (
                    <List.Item className={`${message.type === 'user' ? 'justify-end' : 'justify-start'} mb-4`}>
                      <div className={`flex items-start max-w-[80%] ${message.type === 'user' ? 'flex-row-reverse' : ''}`}>
                        {message.type === 'user' ? (
                          <UserOutlined className="text-blue-500 mr-2 mt-1" />
                        ) : (
                          <RobotOutlined className="text-green-500 mr-2 mt-1" />
                        )}
                        <div className={`${message.type === 'user' ? 'bg-blue-500 text-white rounded-tr-none' : 'bg-gray-200 rounded-tl-none'} px-4 py-2 rounded-lg`}>
                          <Paragraph className="m-0 whitespace-pre-wrap">{message.content}</Paragraph>
                          <Text className="block text-xs opacity-70 mt-1">{message.timestamp}</Text>
                        </div>
                      </div>
                    </List.Item>
                  )}
                />
              )}
              <div ref={messagesEndRef} />
            </div>
            
            {/* 输入区域 */}
            <div className="border-t p-4">
              <div className="flex gap-2">
                <TextArea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="请输入您的问题..."
                  rows={3}
                  onPressEnter={(e) => {
                    if (e.ctrlKey || e.metaKey) {
                      sendMessage()
                    }
                  }}
                />
                <Button
                  type="primary"
                  icon={isLoading ? <LoadingOutlined /> : <SendOutlined />}
                  onClick={sendMessage}
                  disabled={!inputValue.trim() || isLoading}
                  className="self-end"
                />
              </div>
              <Text type="secondary" className="text-xs mt-2 block">
                {isLoading ? '正在思考...' : '按 Ctrl+Enter 或点击发送按钮'}
              </Text>
            </div>
          </Card>
        </div>
        
        {/* 功能说明卡片 */}
        <Card title="功能说明" className="mt-6">
          <List
            dataSource={[
              '支持错别字纠错、古诗词赏析、汉字拼音、成语学习等语文相关问题',
              '聊天记录会保存在本地，最多保存10条历史记录',
              '您可以随时新建聊天或删除历史记录',
              '输入框支持多行文本，按 Ctrl+Enter 快速发送消息',
              '点击"学习分析"按钮获取基于您学习数据的个性化建议'
            ]}
            renderItem={(item, index) => (
              <List.Item>
                <CheckCircleOutlined className="text-green-500 mr-2" />
                <Text>{item}</Text>
              </List.Item>
            )}
          />
        </Card>
      </div>
    </div>
  )
}

export default Chat