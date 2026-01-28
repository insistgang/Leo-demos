import React, { useState, useEffect } from 'react'
import { Card, Input, Button, message, List, Typography, Divider, Radio, Checkbox, Spin, Empty } from 'antd'
import { BookOutlined, CheckCircleOutlined, CloseCircleOutlined, ReloadOutlined } from '@ant-design/icons'
import questionService from '../services/questionService'
import { studyRecordService } from '../services/studyRecordService'
import { mockLiteratureQuestions } from '../services/mockLiteratureQuestions'
import './Literature.css'

const { TextArea } = Input
const { Title, Text, Paragraph } = Typography

const Literature = () => {
  const [questions, setQuestions] = useState([])
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
  const [userAnswers, setUserAnswers] = useState({})
  const [isCompleted, setIsCompleted] = useState(false)
  const [score, setScore] = useState(0)
  const [history, setHistory] = useState([])
  const [isLearningStarted, setIsLearningStarted] = useState(false)
  const [showExplanation, setShowExplanation] = useState(false)
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [submittedAnswers, setSubmittedAnswers] = useState(new Set())

  // 初始化加载
  useEffect(() => {
    loadHistory()
  }, [])

  
  // 从数据库加载历史记录
  const loadHistory = async () => {
    try {
      const result = await studyRecordService.getStudyRecords({
        type: 'literature',
        pageSize: 10
      })
      if (result && result.data) {
        setHistory(result.data)
      }
    } catch (error) {
      console.error('加载历史记录失败:', error)
      // 从本地存储加载备用数据
      const savedHistory = localStorage.getItem('literatureHistory')
      if (savedHistory) {
        setHistory(JSON.parse(savedHistory))
      }
    }
  }

  // 保存学习记录到数据库
  const saveStudyRecord = async (questionId, isCorrect, score) => {
    const question = questions[currentQuestionIndex]

    console.log('\n=== Literature.jsx - 准备保存学习记录 ===');
    console.log('题目ID:', questionId);
    console.log('是否正确:', isCorrect);
    console.log('得分:', score);
    console.log('题目内容:', question);
    console.log('=====================================\n');

    try {
      const recordData = {
        type: 'literature',
        questionId: questionId,
        score: score,
        totalQuestions: 1,
        correctAnswers: isCorrect ? 1 : 0,
        duration: 30, // 假设每题30秒
        content: {
          question: question.question,
          type: question.type
        }
      };

      console.log('发送的数据:', JSON.stringify(recordData, null, 2));

      const result = await studyRecordService.addStudyRecord(recordData);
      console.log('保存结果:', result);

      // 重新加载历史记录
      loadHistory()
    } catch (error) {
      console.error('保存学习记录失败，保存到本地:', error)
      // 保存到本地存储作为备份
      const record = {
        id: Date.now(),
        type: 'literature',
        title: question.question.substring(0, 50) + '...',
        score: score,
        studyDate: new Date().toLocaleDateString('zh-CN'),
        isCorrect: isCorrect
      }
      const newHistory = [record, ...history.slice(0, 9)]
      setHistory(newHistory)
      localStorage.setItem('literatureHistory', JSON.stringify(newHistory))
    }
  }

  // 开始学习
  const startLearning = async () => {
    setLoading(true)
    setIsLearningStarted(true)

    try {
      // 尝试从API获取题目
      const fetchedQuestions = await questionService.getQuestionsByType('literature', { count: 10 })
      if (fetchedQuestions && fetchedQuestions.length > 0) {
        setQuestions(fetchedQuestions)
        setCurrentQuestionIndex(0)
        resetQuestionState()
      } else {
        // 如果API没有数据，使用模拟数据
        console.log('使用模拟古诗词题目数据')
        setQuestions(mockLiteratureQuestions)
        setCurrentQuestionIndex(0)
        resetQuestionState()
      }
    } catch (error) {
      console.error('从API加载题目失败，使用模拟数据:', error)
      // 使用模拟数据
      setQuestions(mockLiteratureQuestions)
      setCurrentQuestionIndex(0)
      resetQuestionState()
    } finally {
      setLoading(false)
    }
  }

  // 重置问题状态
  const resetQuestionState = () => {
    setUserAnswers({})
    setIsCompleted(false)
    setScore(0)
    setShowExplanation(false)
    setResults([])
  }

  // 处理答案变化
  const handleAnswerChange = (value, questionId) => {
    setUserAnswers(prev => ({
      ...prev,
      [questionId]: value
    }))
  }

  // 处理多选答案变化
  const handleMultiSelectChange = (checkedValues, questionId) => {
    setUserAnswers(prev => ({
      ...prev,
      [questionId]: checkedValues
    }))
  }

  // 提交答案
  const submitAnswer = async () => {
    if (!questions[currentQuestionIndex]) return

    const question = questions[currentQuestionIndex]
    const userAnswer = userAnswers[question.id]
    let isCorrect = false

    console.log('\n=== 提交答案调试信息 ===')
    console.log('题目ID:', question.id)
    console.log('用户答案:', userAnswer, '(类型:', typeof userAnswer, ')')
    console.log('正确答案:', question.correctAnswer, '(类型:', typeof question.correctAnswer, ')')
    console.log('题目类型:', question.type)
    console.log('========================\n')

    // 根据题型判断答案是否正确
    if (question.type === 'single_choice') {
      isCorrect = userAnswer === question.correctAnswer
    } else if (question.type === 'multiple_choice') {
      isCorrect = userAnswer &&
                  userAnswer.length === question.correctAnswer.length &&
                  userAnswer.every(answer => question.correctAnswer.includes(answer))
    } else if (question.type === 'blank_filling') {
      // 填空题答案可能是一个字符串或数组
      if (Array.isArray(question.correctAnswer)) {
        isCorrect = userAnswer && question.correctAnswer.some(correct =>
          userAnswer.trim().toLowerCase() === correct.toLowerCase()
        )
      } else {
        isCorrect = userAnswer &&
                  userAnswer.trim().toLowerCase() === question.correctAnswer.toLowerCase()
      }
    }

    // 计算得分（每题10分）
    const points = isCorrect ? 10 : 0
    const newScore = score + points

    // 更新结果
    const newResult = {
      questionId: question.id,
      question: question.question,
      userAnswer,
      correctAnswer: question.correctAnswer,
      isCorrect,
      explanation: question.explanation
    }

    setResults(prev => [...prev, newResult])
    setScore(newScore)
    setIsCompleted(true)
    setShowExplanation(true) // 自动显示解析
    setSubmittedAnswers(prev => new Set(prev).add(question.id))

    // 保存到数据库
    await saveStudyRecord(question.id, isCorrect, points)

    // 显示反馈
    if (isCorrect) {
      message.success('回答正确！获得10分')
    } else {
      message.error('回答错误，请查看解析')
    }

    // 2秒后自动跳转到下一题
    window.autoNextTimer = setTimeout(() => {
      if (currentQuestionIndex < questions.length - 1) {
        setCurrentQuestionIndex(prev => prev + 1)
        setIsCompleted(false)
        setShowExplanation(false)
      } else {
        // 所有问题已完成
        message.info('您已完成所有题目！')
        setIsLearningStarted(false)
        setQuestions([])
      }
    }, 2000)
  }

  // 下一题
  const nextQuestion = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(prev => prev + 1)
      resetQuestionState()
    } else {
      // 所有问题已完成
      message.info('您已完成所有题目！')
      setIsLearningStarted(false)
      setQuestions([])
    }
  }

  // 重新开始
  const restart = () => {
    setCurrentQuestionIndex(0)
    setScore(0)
    setResults([])
    setSubmittedAnswers(new Set())
    setIsLearningStarted(false)
    setQuestions([])
    resetQuestionState()
  }

  // 渲染题目
  const renderQuestion = () => {
    if (!questions[currentQuestionIndex]) return null

    const question = questions[currentQuestionIndex]
    const hasAnswered = submittedAnswers.has(question.id)

    switch (question.type) {
      case 'single_choice':
        return (
          <div className="single-choice-question">
            <Paragraph className="question-text">{question.question}</Paragraph>
            <Radio.Group
              value={userAnswers[question.id]}
              onChange={(e) => handleAnswerChange(e.target.value, question.id)}
              disabled={hasAnswered}
            >
              {question.options.map((option, index) => (
                <Radio key={index} value={index} style={{ display: 'block', marginBottom: 8 }}>
                  {option}
                </Radio>
              ))}
            </Radio.Group>
          </div>
        )

      case 'multiple_choice':
        return (
          <div className="multiple-choice-question">
            <Paragraph className="question-text">{question.question}</Paragraph>
            <Checkbox.Group
              value={userAnswers[question.id] || []}
              onChange={(checkedValues) => handleMultiSelectChange(checkedValues, question.id)}
              disabled={hasAnswered}
            >
              {question.options.map((option, index) => (
                <Checkbox key={index} value={index} style={{ display: 'block', marginBottom: 8 }}>
                  {option}
                </Checkbox>
              ))}
            </Checkbox.Group>
          </div>
        )

      case 'blank_filling':
        // 处理填空题
        const questionText = question.question
        const parts = questionText.split(/___+/)
        const blankCount = questionText.match(/___+/g)?.length || 1

        return (
          <div className="blank-filling-question">
            <div className="question-content">
              {parts.map((part, index) => (
                <React.Fragment key={index}>
                  <span>{part}</span>
                  {index < blankCount && (
                    <Input
                      style={{ width: 150, margin: '0 8px' }}
                      placeholder="请填写答案"
                      value={userAnswers[`${question.id}_${index}`] || ''}
                      onChange={(e) => {
                        const newAnswers = { ...userAnswers }
                        newAnswers[`${question.id}_${index}`] = e.target.value
                        // 合并所有空格的答案
                        const allAnswers = []
                        for (let i = 0; i < blankCount; i++) {
                          if (newAnswers[`${question.id}_${i}`]) {
                            allAnswers.push(newAnswers[`${question.id}_${i}`])
                          }
                        }
                        newAnswers[question.id] = allAnswers.join('、')
                        setUserAnswers(newAnswers)
                      }}
                      disabled={hasAnswered}
                    />
                  )}
                </React.Fragment>
              ))}
            </div>
          </div>
        )

      default:
        return <Text type="secondary">未知的题型</Text>
    }
  }

  // 渲染解析
  const renderExplanation = () => {
    const currentResult = results.find(r => r.questionId === questions[currentQuestionIndex]?.id)
    if (!currentResult || !showExplanation) return null

    return (
      <Card className="explanation-card mt-4" style={{ backgroundColor: '#f6f8fa' }}>
        <div className="explanation-content">
          <div className="answer-status">
            {currentResult.isCorrect ? (
              <Text type="success">
                <CheckCircleOutlined /> 回答正确！
              </Text>
            ) : (
              <Text type="danger">
                <CloseCircleOutlined /> 回答错误
              </Text>
            )}
          </div>
          <Divider style={{ margin: '12px 0' }} />
          <div className="correct-answer">
            <Text strong>正确答案：</Text>
            <Text>
              {Array.isArray(currentResult.correctAnswer)
                ? currentResult.correctAnswer.map((idx, index) => (
                    <span key={idx}>
                      {questions[currentQuestionIndex].options[idx]}
                      {index < currentResult.correctAnswer.length - 1 ? '、' : ''}
                    </span>
                  ))
                : typeof currentResult.correctAnswer === 'number'
                ? questions[currentQuestionIndex].options[currentResult.correctAnswer]
                : currentResult.correctAnswer}
            </Text>
          </div>
          {currentResult.explanation && (
            <>
              <Divider style={{ margin: '12px 0' }} />
              <div className="explanation-text">
                <Text strong>解析：</Text>
                <Paragraph>{currentResult.explanation}</Paragraph>
              </div>
            </>
          )}
        </div>
      </Card>
    )
  }

  // 渲染历史记录
  const renderHistory = () => {
    return (
      <Card title="练习历史" className="history-card">
        {history.length > 0 ? (
          <List
            dataSource={history}
            renderItem={item => (
              <List.Item>
                <List.Item.Meta
                  title={item.title || item.content?.question?.substring(0, 50) + '...'}
                  description={`${item.studyDate} | 得分：${item.score}分`}
                />
              </List.Item>
            )}
            pagination={{ pageSize: 5 }}
          />
        ) : (
          <Empty description="暂无练习记录" image={Empty.PRESENTED_IMAGE_SIMPLE} />
        )}
      </Card>
    )
  }

  // 渲染进度
  const renderProgress = () => {
    if (!isLearningStarted || questions.length === 0) return null

    return (
      <Card className="progress-card mb-4" size="small">
        <div className="progress-info">
          <Text>题目进度：{currentQuestionIndex + 1} / {questions.length}</Text>
          <Text className="ml-4">当前得分：{score}分</Text>
        </div>
      </Card>
    )
  }

  return (
    <div className="literature-container min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center">
            <BookOutlined className="text-purple-600 mr-2" />
            <Title level={2} className="mb-0">古诗词学习</Title>
          </div>
          {isLearningStarted && (
            <Button
              icon={<ReloadOutlined />}
              onClick={restart}
            >
              重新开始
            </Button>
          )}
        </div>

        {renderProgress()}

        <div className="main-content">
          <Card className="question-card mb-6">
            {loading ? (
              <div className="text-center py-12">
                <Spin size="large" />
                <div className="mt-4">
                  <Text type="secondary">正在加载题目...</Text>
                </div>
              </div>
            ) : !isLearningStarted ? (
              <div className="text-center py-12">
                <BookOutlined className="text-purple-500 text-6xl mb-4" />
                <Title level={4} className="mb-4">探索中华文化的瑰宝</Title>
                <Text className="block mb-6 text-gray-600">通过古诗词练习，提升语文基础知识水平</Text>
                <Button
                  type="primary"
                  size="large"
                  icon={<BookOutlined />}
                  onClick={startLearning}
                  className="bg-purple-600 hover:bg-purple-700"
                >
                  开始练习
                </Button>
              </div>
            ) : questions[currentQuestionIndex] ? (
              <>
                <div className="question-header mb-4">
                  <Text type="secondary" className="question-source">
                    {questions[currentQuestionIndex].source}
                  </Text>
                </div>

                {renderQuestion()}

                <Divider />

                {isCompleted ? (
                  <>
                    <div className="result-section">
                      <div className="action-buttons">
                        <Text type="secondary" style={{ marginRight: 10 }}>
                          2秒后自动跳转到下一题...
                        </Text>
                        <Button
                          type="primary"
                          onClick={() => {
                            clearTimeout(window.autoNextTimer)
                            nextQuestion()
                          }}
                        >
                          立即下一题
                        </Button>
                      </div>
                    </div>
                    {showExplanation ? renderExplanation() : null}
                  </>
                ) : (
                  <Button
                    type="primary"
                    onClick={submitAnswer}
                    className="submit-btn"
                    disabled={userAnswers[questions[currentQuestionIndex]?.id] === undefined || userAnswers[questions[currentQuestionIndex]?.id] === null}
                  >
                    提交答案
                  </Button>
                )}
              </>
            ) : (
              <Empty description="暂无题目" />
            )}
          </Card>

          {renderHistory()}
        </div>
      </div>
    </div>
  )
}

export default Literature