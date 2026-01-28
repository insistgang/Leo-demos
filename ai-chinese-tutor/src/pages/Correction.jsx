import React, { useState, useRef, useEffect } from 'react'
import { Card, Radio, Button, message, Typography, List, Tag, Divider, Alert, Spin } from 'antd'
import { HighlightOutlined, CheckCircleOutlined, CloseCircleOutlined, BookOutlined, EditOutlined, RotateLeftOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { questionService } from '../services/questionService'
import { studyRecordService } from '../services/studyRecordService'

const { Title, Text, Paragraph } = Typography

const Correction = () => {
  const navigate = useNavigate()
  const [isLearningStarted, setIsLearningStarted] = useState(false)
  const [questions, setQuestions] = useState([])
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
  const [selectedAnswer, setSelectedAnswer] = useState(null)
  const [correctionResult, setCorrectionResult] = useState(null)
  const [score, setScore] = useState(0)
  const [completedQuestions, setCompletedQuestions] = useState([])
  const [showHistory, setShowHistory] = useState(false)
  const [practiceHistory, setPracticeHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [loadingQuestions, setLoadingQuestions] = useState(false)

  // 加载历史记录
  useEffect(() => {
    loadPracticeHistory()
  }, [])

  // 加载题目
  const loadQuestions = async () => {
    setLoadingQuestions(true)
    try {
      const questionsData = await questionService.getQuestionsByType('correction', { count: 10 })
      if (questionsData && questionsData.length > 0) {
        setQuestions(questionsData)
      } else {
        message.warning('暂无错别字纠错题目')
      }
    } catch (error) {
      console.error('加载题目失败:', error)
      message.error('加载题目失败，请稍后重试')
    } finally {
      setLoadingQuestions(false)
    }
  }

  // 加载练习历史
  const loadPracticeHistory = async () => {
    try {
      const result = await studyRecordService.getStudyRecords({ type: 'correction' })
      if (result && result.data) {
        setPracticeHistory(result.data)
      }
    } catch (error) {
      console.error('加载练习历史失败:', error)
    }
  }

  // 开始学习
  const startLearning = async () => {
    setLoadingQuestions(true)
    await loadQuestions()
    setLoadingQuestions(false)

    if (questions.length === 0) return

    setIsLearningStarted(true)
    setCurrentQuestionIndex(0)
    setSelectedAnswer(null)
    setCorrectionResult(null)
  }

  // 提交答案
  const submitAnswer = async () => {
    if (selectedAnswer === null) {
      message.warning('请选择一个答案')
      return
    }

    const currentQuestion = questions[currentQuestionIndex]
    const isCorrect = selectedAnswer === currentQuestion.correctAnswer

    // 更新分数
    const newScore = isCorrect ? score + 10 : score
    setScore(newScore)

    // 显示结果
    setCorrectionResult({
      isCorrect,
      userAnswer: currentQuestion.options[selectedAnswer],
      correctAnswer: currentQuestion.options[currentQuestion.correctAnswer],
      explanation: currentQuestion.explanation
    })

    // 保存到数据库
    try {
      const recordData = {
        type: 'correction',
        title: '错别字纠错练习',
        content: {
          questionId: currentQuestion.id,
          question: currentQuestion.question,
          userAnswer: currentQuestion.options[selectedAnswer],
          correctAnswer: currentQuestion.options[currentQuestion.correctAnswer],
          isCorrect
        },
        score: newScore,
        totalQuestions: 1,
        correctAnswers: isCorrect ? 1 : 0,
        duration: 30 // 假设每题30秒
      }

      await studyRecordService.addStudyRecord(recordData)
      message.success('答题记录已保存')
    } catch (error) {
      console.error('保存答题记录失败:', error)
      message.error('保存答题记录失败')
    }

    // 标记为已完成
    setCompletedQuestions([...completedQuestions, { id: currentQuestion.id, correct: isCorrect }])
  }

  // 下一题
  const nextQuestion = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1)
      setSelectedAnswer(null)
      setCorrectionResult(null)
    } else {
      // 完成所有题目
      completeAllQuestions()
    }
  }

  // 完成所有题目
  const completeAllQuestions = async () => {
    message.success('恭喜您完成了所有错别字纠错练习！')

    // 计算正确答案数
    const correctCount = completedQuestions.filter(q => q.correct).length

    // 记录练习结果
    try {
      const recordData = {
        type: 'correction',
        title: '错别字纠错练习完成',
        content: {
          totalQuestions: questions.length,
          finalScore: score,
          accuracy: questions.length > 0 ? Math.round((correctCount / questions.length) * 100) : 0
        },
        score: score,
        totalQuestions: questions.length,
        correctAnswers: correctCount,
        duration: questions.length * 30
      }

      await studyRecordService.addStudyRecord(recordData)
    } catch (error) {
      console.error('保存练习完成记录失败:', error)
    }

    setShowHistory(true)
    await loadPracticeHistory()
  }

  // 重新开始
  const restartPractice = () => {
    setIsLearningStarted(false)
    setQuestions([])
    setCurrentQuestionIndex(0)
    setSelectedAnswer(null)
    setCorrectionResult(null)
    setScore(0)
    setCompletedQuestions([])
    setShowHistory(false)
  }

  // 渲染纠错结果
  const renderCorrectionResult = () => {
    if (!correctionResult) return null

    return (
      <div className="mt-4">
        <Title level={5} className="mb-2">答题结果</Title>
        <div className="mb-3">
          <Text strong>您的答案：</Text>
          <div className="p-2 border rounded bg-gray-50 mt-1">
            {correctionResult.userAnswer}
          </div>
        </div>

        <div className="mb-3">
          <Text strong>正确答案：</Text>
          <div className="p-2 border rounded bg-green-50 mt-1">
            {correctionResult.correctAnswer}
          </div>
        </div>

        {correctionResult.explanation && (
          <div className="mb-3">
            <Text strong>解析：</Text>
            <div className="p-2 border rounded bg-blue-50 mt-1">
              <Text>{correctionResult.explanation}</Text>
            </div>
          </div>
        )}

        {correctionResult.isCorrect ? (
          <Alert type="success" message="太棒了！您的答案完全正确！" showIcon />
        ) : (
          <Alert type="error" message="答案错误，请继续努力！" showIcon />
        )}
      </div>
    )
  }

  // 渲染练习历史
  const renderPracticeHistory = () => {
    return (
      <Card title="练习历史" className="mt-4">
        <List
          dataSource={practiceHistory}
          renderItem={(record) => (
            <List.Item>
              <List.Item.Meta
                title={
                  <div className="flex justify-between items-center">
                    <span>{record.title}</span>
                    <span className="text-blue-600">得分：{record.score}</span>
                  </div>
                }
                description={
                  <div className="flex justify-between items-center">
                    <span>完成题目数：{record.totalQuestions} | 正确：{record.correctAnswers}</span>
                    <span>{record.studyDate}</span>
                  </div>
                }
              />
            </List.Item>
          )}
          locale={{ emptyText: '暂无练习历史' }}
        />
      </Card>
    )
  }

  // 获取当前题目
  const currentQuestion = questions[currentQuestionIndex]

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center mb-6">
          <HighlightOutlined className="text-red-600 mr-2" />
          <Title level={2}>错别字纠错练习</Title>
        </div>
        
        <Card className="mb-6">
          <div className="flex justify-between items-center mb-4">
            <Title level={4}>练习进度</Title>
            <div>
              <Text strong className="mr-4">当前得分：{score}</Text>
              {isLearningStarted && (
                <Text type="secondary">
                  题目 {currentQuestionIndex + 1} / {questions.length}
                </Text>
              )}
            </div>
          </div>

          {loadingQuestions ? (
            <div className="text-center py-10">
              <Spin size="large" />
              <Title level={4} className="mt-4">加载题目中...</Title>
            </div>
          ) : !isLearningStarted && !showHistory ? (
            <div className="text-center py-10">
              <HighlightOutlined className="text-red-500 text-6xl mb-4" />
              <Title level={4} className="mb-4">提高语文水平的重要技能</Title>
              <Text className="block mb-6 text-gray-600">错别字纠错是学习语文的基础技能，通过练习可以提高文字表达的准确性</Text>
              <Button
                type="primary"
                size="large"
                icon={<EditOutlined />}
                onClick={startLearning}
                className="bg-blue-600 hover:bg-blue-700"
              >
                开始学习
              </Button>
            </div>
          ) : showHistory ? (
            <div>
              <Alert type="success" message="恭喜您完成了所有错别字纠错练习！" showIcon className="mb-4" />
              {renderPracticeHistory()}
              <Button
                type="primary"
                block
                icon={<RotateLeftOutlined />}
                onClick={restartPractice}
                className="mt-4"
              >
                重新开始练习
              </Button>
            </div>
          ) : currentQuestion ? (
            <div>
              <div className="mb-4">
                <Title level={5}>题目 {currentQuestionIndex + 1}：</Title>
                <div className="p-4 bg-gray-50 rounded mb-4">
                  <Paragraph>{currentQuestion.question}</Paragraph>
                </div>

                <div className="mb-4">
                  <Radio.Group
                    value={selectedAnswer}
                    onChange={(e) => setSelectedAnswer(e.target.value)}
                    disabled={correctionResult !== null}
                  >
                    {currentQuestion.options.map((option, index) => (
                      <Radio key={index} value={index} className="block mb-2">
                        {option}
                      </Radio>
                    ))}
                  </Radio.Group>
                </div>

                <div className="flex space-x-2">
                  <Button
                    type="primary"
                    onClick={submitAnswer}
                    disabled={selectedAnswer === null || correctionResult !== null}
                  >
                    提交答案
                  </Button>
                  {correctionResult === null && (
                    <Button onClick={() => setSelectedAnswer(null)}>清空选择</Button>
                  )}
                </div>
              </div>

              {correctionResult && renderCorrectionResult()}

              {correctionResult && (
                <div className="mt-4 flex justify-end">
                  <Button type="primary" onClick={nextQuestion}>
                    {currentQuestionIndex < questions.length - 1 ? '下一题' : '完成练习'}
                  </Button>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-10">
              <Text type="secondary">暂无题目</Text>
            </div>
          )}
        </Card>
        
        {/* 学习提示卡片 */}
        <Card title="学习提示" type="inner">
          <List
            dataSource={[
              '注意区分同音字和形近字',
              '理解词语的含义有助于正确使用',
              '多阅读优秀作品可以提高对文字的敏感度',
              '练习时注重分析错误原因，而不仅仅是纠正' 
            ]}
            renderItem={(tip, index) => (
              <List.Item>
                <CheckCircleOutlined className="text-green-500 mr-2" />
                <Text>{tip}</Text>
              </List.Item>
            )}
          />
        </Card>
        
        {/* 相关推荐 */}
        <div className="mt-6">
          <Divider>相关推荐</Divider>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card
              hoverable
              className="cursor-pointer"
              onClick={() => navigate('/literature')}
            >
              <div className="flex items-center">
                <BookOutlined className="text-purple-600 mr-2" />
                <Text strong>古诗词赏析</Text>
              </div>
            </Card>
            <Card
              hoverable
              className="cursor-pointer"
              onClick={() => navigate('/vocabulary')}
            >
              <div className="flex items-center">
                <BookOutlined className="text-blue-600 mr-2" />
                <Text strong>词汇学习</Text>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Correction