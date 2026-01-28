import React, { useState, useEffect } from 'react'
import { Card, Button, Input, List, Radio, message, Divider, Progress, Statistic, Row, Col, Spin, Alert } from 'antd'
import { BookOutlined, CheckOutlined, ArrowRightOutlined, ReloadOutlined, CloseCircleOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { studyRecordService } from '../services/studyRecordService'
import { questionService } from '../services/questionService'

const Pinyin = () => {
  const navigate = useNavigate()
  const [isLearningStarted, setIsLearningStarted] = useState(false)
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
  const [selectedOption, setSelectedOption] = useState(null)
  const [showFeedback, setShowFeedback] = useState(false)
  const [isAnswerCorrect, setIsAnswerCorrect] = useState(false)
  const [practiceResult, setPracticeResult] = useState({ correct: 0, total: 0 })
  const [exerciseCompleted, setExerciseCompleted] = useState(false)
  const [progress, setProgress] = useState(0)
  const [answerHistory, setAnswerHistory] = useState([])
  const [showHistory, setShowHistory] = useState(false)
  const [questions, setQuestions] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  // 开始学习 - 从数据库加载题目
  const startLearning = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const questionsData = await questionService.getQuestionsByType('pinyin', { count: 10 })

      if (questionsData && questionsData.length > 0) {
        setQuestions(questionsData)
        setCurrentQuestionIndex(0)
        setPracticeResult({ correct: 0, total: questionsData.length })
        setProgress(0)
        setIsLearningStarted(true)
        setExerciseCompleted(false)
        setAnswerHistory([])
        setSelectedOption(null)
        setShowFeedback(false)
        setIsAnswerCorrect(false)
      } else {
        setError('没有找到拼音练习题目，请稍后再试')
        message.error('没有找到拼音练习题目')
      }
    } catch (error) {
      console.error('加载题目失败:', error)
      setError('加载题目失败，请稍后再试')
      message.error('加载题目失败，请稍后再试')
    } finally {
      setIsLoading(false)
    }
  }

  // 处理选项选择
  const handleOptionSelect = async (e) => {
    const selectedValue = e.target.value
    setSelectedOption(selectedValue)

    const currentQuestion = questions[currentQuestionIndex]
    const isCorrect = parseInt(selectedValue) === currentQuestion.correctAnswer
    setIsAnswerCorrect(isCorrect)
    setShowFeedback(true)

    // 计算得分 (每题10分)
    const points = isCorrect ? 10 : 0

    // 更新练习结果
    if (isCorrect) {
      setPracticeResult(prev => ({
        ...prev,
        correct: prev.correct + 1
      }))
      message.success('回答正确！+10分')
    } else {
      message.error('回答错误')
    }

    // 记录答案历史
    const newHistory = [...answerHistory]
    newHistory[currentQuestionIndex] = {
      questionId: currentQuestion.id,
      question: currentQuestion.question,
      selectedOption: parseInt(selectedValue),
      correctAnswer: currentQuestion.correctAnswer,
      isCorrect: isCorrect,
      points: points,
      explanation: currentQuestion.explanation
    }
    setAnswerHistory(newHistory)

    // 保存答题结果到数据库
    try {
      await saveAnswerResult(currentQuestion.id, selectedValue, isCorrect, points)
    } catch (error) {
      console.error('保存答案失败:', error)
    }
  }

  // 保存单个答题结果
  const saveAnswerResult = async (questionId, selectedAnswer, isCorrect, points) => {
    try {
      const recordData = {
        type: 'pinyin',
        content: {
          questionId,
          selectedAnswer,
          isCorrect,
          points
        },
        duration: 0, // TODO: 计算单题用时
        score: points,
        totalQuestions: 1,
        correctAnswers: isCorrect ? 1 : 0
      }

      await studyRecordService.addStudyRecord(recordData)
    } catch (error) {
      console.error('保存答题记录出错:', error)
    }
  }

  // 下一题
  const goToNextQuestion = () => {
    if (currentQuestionIndex < questions.length - 1) {
      const nextIndex = currentQuestionIndex + 1
      setCurrentQuestionIndex(nextIndex)
      setSelectedOption(null)
      setShowFeedback(false)
      setIsAnswerCorrect(false)
      setProgress(Math.round(((nextIndex + 1) / questions.length) * 100))
    } else {
      // 练习完成
      setExerciseCompleted(true)
      setProgress(100)

      // 保存整体练习结果到数据库
      savePracticeResult()
    }
  }

  // 保存练习结果到数据库
  const savePracticeResult = async () => {
    try {
      const recordData = {
        type: 'pinyin',
        content: {
          totalQuestions: practiceResult.total,
          correctAnswers: practiceResult.correct,
          accuracy: Math.round((practiceResult.correct / practiceResult.total) * 100),
          answerHistory: answerHistory
        },
        duration: 0, // TODO: 计算实际用时
        score: practiceResult.correct * 10, // 每题10分
        totalQuestions: practiceResult.total,
        correctAnswers: practiceResult.correct
      }

      await studyRecordService.addStudyRecord(recordData)
      console.log('拼音练习记录保存成功')
    } catch (error) {
      console.error('保存拼音练习记录出错:', error)
    }
  }

  // 重新开始
  const restartLearning = () => {
    setIsLearningStarted(false)
    setExerciseCompleted(false)
    setCurrentQuestionIndex(0)
    setPracticeResult({ correct: 0, total: 0 })
    setProgress(0)
    setAnswerHistory([])
    setSelectedOption(null)
    setShowFeedback(false)
    setIsAnswerCorrect(false)
    setShowHistory(false)
    setQuestions([])
    setError(null)
  }

  // 返回主页
  const returnToHome = () => {
    navigate('/home')
  }

  // 渲染开始学习界面
  const renderStartScreen = () => (
    <Card title="汉字拼音练习" className="pinyin-start-card">
      <div className="pinyin-start-content">
        <h3>练习汉字的拼音读音</h3>
        <p>通过选择正确的拼音，提升汉字发音识别能力</p>
        <p>每道题10分，答对后立即得分</p>

        {error && (
          <Alert
            message={error}
            type="error"
            style={{ marginBottom: 20 }}
          />
        )}

        <Button
          type="primary"
          size="large"
          onClick={startLearning}
          icon={<BookOutlined />}
          className="start-btn"
          loading={isLoading}
        >
          开始练习
        </Button>
      </div>
    </Card>
  )

  // 渲染练习界面
  const renderLearningScreen = () => {
    if (isLoading || questions.length === 0) {
      return (
        <Card className="loading-card">
          <div className="loading-content">
            <Spin size="large" />
            <p style={{ marginTop: 16 }}>正在加载练习内容...</p>
          </div>
        </Card>
      )
    }

    const currentQuestion = questions[currentQuestionIndex]

    return (
      <>
        <div className="progress-section">
          <Row gutter={16}>
            <Col span={16}>
              <Progress percent={progress} status="active" showInfo />
            </Col>
            <Col span={8}>
              <Statistic
                title="正确率"
                value={practiceResult.correct > 0 ? Math.round((practiceResult.correct / (currentQuestionIndex + 1)) * 100) : 0}
                suffix="%"
              />
            </Col>
          </Row>
        </div>

        <Card title={`题目 ${currentQuestionIndex + 1}/${questions.length}`} className="pinyin-learning-card">
          <div className="question-display">
            <h3>{currentQuestion.question}</h3>
          </div>

          <Divider>
            请选择正确答案
          </Divider>

          <Radio.Group onChange={handleOptionSelect} value={selectedOption} className="options-group">
            <List
              grid={{ gutter: 16, column: 1 }}
              dataSource={currentQuestion.options}
              renderItem={(option, index) => (
                <List.Item>
                  <Radio
                    value={index}
                    className={`option-radio ${selectedOption === index ? 'selected' : ''}`}
                    disabled={showFeedback}
                  >
                    {option}
                  </Radio>
                </List.Item>
              )}
            />
          </Radio.Group>

          {showFeedback && (
            <div className={`answer-feedback ${isAnswerCorrect ? 'correct' : 'incorrect'}`} style={{ marginBottom: 16 }}>
              {isAnswerCorrect ? (
                <Alert
                  message="回答正确！+10分"
                  type="success"
                  icon={<CheckOutlined />}
                  showIcon
                />
              ) : (
                <Alert
                  message="回答错误！"
                  description={`正确答案是：${currentQuestion.options[currentQuestion.correctAnswer]}`}
                  type="error"
                  icon={<CloseCircleOutlined />}
                  showIcon
                />
              )}
            </div>
          )}

          {showFeedback && currentQuestion.explanation && (
            <Alert
              message="解析"
              description={currentQuestion.explanation}
              type="info"
              style={{ marginBottom: 16 }}
            />
          )}

          <div className="action-buttons">
            <Button
              type="primary"
              onClick={goToNextQuestion}
              icon={<ArrowRightOutlined />}
              disabled={!showFeedback}
              className="next-btn"
              size="large"
            >
              {currentQuestionIndex < questions.length - 1 ? '下一题' : '完成练习'}
            </Button>
          </div>
        </Card>
      </>
    )
  }

  // 渲染练习完成界面
  const renderCompletedScreen = () => (
    <Card title="练习完成" className="pinyin-completed-card">
      <div className="completed-content">
        <div className="result-summary">
          <h3>恭喜你完成了所有练习！</h3>
          <div className="result-stats">
            <Row gutter={16}>
              <Col span={8}>
                <Statistic title="总题数" value={practiceResult.total} />
              </Col>
              <Col span={8}>
                <Statistic title="正确数" value={practiceResult.correct} />
              </Col>
              <Col span={8}>
                <Statistic
                  title="正确率"
                  value={Math.round((practiceResult.correct / practiceResult.total) * 100)}
                  suffix="%"
                />
              </Col>
            </Row>
            <div style={{ marginTop: 20, textAlign: 'center' }}>
              <Statistic
                title="总得分"
                value={practiceResult.correct * 10}
                suffix="分"
                valueStyle={{ color: '#3f8600', fontSize: '36px' }}
              />
            </div>
          </div>
        </div>

        <Button
          type="primary"
          onClick={() => setShowHistory(!showHistory)}
          className="history-btn"
          style={{ marginBottom: 20 }}
        >
          {showHistory ? '隐藏答案历史' : '查看答案历史'}
        </Button>

        {showHistory && (
          <div className="answer-history">
            <h4>答题历史</h4>
            <List
              dataSource={answerHistory}
              renderItem={(item, index) => (
                <List.Item className={`history-item ${item.isCorrect ? 'correct' : 'incorrect'}`}>
                  <List.Item.Meta
                    title={`题目 ${index + 1}: ${item.question.substring(0, 50)}${item.question.length > 50 ? '...' : ''}`}
                    description={
                      <div>
                        <div>
                          <span style={{ color: item.isCorrect ? '#52c41a' : '#ff4d4f' }}>
                            {item.isCorrect ? '✓ 正确' : '✗ 错误'}
                          </span>
                          <span style={{ marginLeft: 10 }}>得分: {item.points}分</span>
                        </div>
                        {!item.isCorrect && (
                          <div style={{ marginTop: 4 }}>
                            <span>你的答案: {item.selectedOption >= 0 ? questions[index].options[item.selectedOption] : '未选择'}</span>
                            <span style={{ marginLeft: 10 }}>
                              正确答案: {questions[index].options[item.correctAnswer]}
                            </span>
                          </div>
                        )}
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </div>
        )}

        <div className="final-actions">
          <Button
            type="primary"
            onClick={restartLearning}
            icon={<ReloadOutlined />}
            className="restart-btn"
            style={{ marginRight: 10 }}
          >
            重新练习
          </Button>
          <Button
            onClick={returnToHome}
            className="home-btn"
          >
            返回主页
          </Button>
        </div>
      </div>
    </Card>
  )

  return (
    <div className="pinyin-container">
      {!isLearningStarted && renderStartScreen()}
      {isLearningStarted && !exerciseCompleted && renderLearningScreen()}
      {isLearningStarted && exerciseCompleted && renderCompletedScreen()}
    </div>
  )
}

export default Pinyin