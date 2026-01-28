import React, { useState, useEffect } from 'react'
import { Card, Button, Radio, Space, Progress, message, Modal, Result, Divider, Row, Col, Statistic, Tag, Spin, Checkbox, Input } from 'antd'
import { ClockCircleOutlined, TrophyOutlined, CheckCircleOutlined, CloseCircleOutlined, ExclamationCircleOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { studyRecordService } from '../services/studyRecordService'
import questionService from '../services/questionService'
import api from '../services/api'

const { confirm } = Modal

const MockExam = () => {
  const navigate = useNavigate()

  // 考试配置
  const [examConfig, setExamConfig] = useState(null)
  const [showConfig, setShowConfig] = useState(true)

  // 考试状态
  const [questions, setQuestions] = useState([])
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
  const [answers, setAnswers] = useState({})
  const [timeLeft, setTimeLeft] = useState(0)
  const [isStarted, setIsStarted] = useState(false)
  const [isFinished, setIsFinished] = useState(false)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showResultDetail, setShowResultDetail] = useState(false)

  // 题目类型选项
  const questionTypes = [
    { value: 'all', label: '全部题型' },
    { value: 'pinyin', label: '拼音' },
    { value: 'vocabulary', label: '词汇' },
    { value: 'literature', label: '古诗词' },
    { value: 'idiom', label: '成语' },
    { value: 'correction', label: '病句修改' },
    { value: 'grammar', label: '语法' },
    { value: 'reading', label: '阅读理解' },
    { value: 'comprehension', label: '综合' }
  ]

  // 考试时长选项
  const durationOptions = [
    { value: 30, label: '30分钟' },
    { value: 60, label: '60分钟' },
    { value: 90, label: '90分钟' },
    { value: 120, label: '120分钟' }
  ]

  // 题目数量选项
  const countOptions = [
    { value: 20, label: '20题' },
    { value: 50, label: '50题' },
    { value: 100, label: '100题' }
  ]

  // 开始考试
  const startExam = async () => {
    if (!examConfig) {
      message.error('请先配置考试参数')
      return
    }

    setLoading(true)
    try {
      const params = {
        count: examConfig.count,
        type: examConfig.type === 'all' ? undefined : examConfig.type
      }

      const questionsData = await questionService.getRandomQuestions(params)

      if (questionsData && questionsData.length > 0) {
        setQuestions(questionsData)
        setTimeLeft(examConfig.duration * 60)
        setIsStarted(true)
        setShowConfig(false)
        setAnswers({})
        setCurrentQuestionIndex(0)
        message.success('考试开始，祝您好运！')
      } else {
        message.error('没有找到符合条件的题目')
      }
    } catch (error) {
      console.error('获取题目失败:', error)
      message.error('获取题目失败，请重试')
    } finally {
      setLoading(false)
    }
  }

  // 提交答案
  const handleAnswer = (questionId, answer) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: answer
    }))
  }

  // 下一题
  const nextQuestion = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(prev => prev + 1)
    }
  }

  // 上一题
  const prevQuestion = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(prev => prev - 1)
    }
  }

  // 交卷确认
  const submitExam = () => {
    const unansweredCount = questions.length - Object.keys(answers).length

    confirm({
      title: '确认交卷',
      icon: <ExclamationCircleOutlined />,
      content: unansweredCount > 0
        ? `您还有 ${unansweredCount} 道题目未作答，确定要交卷吗？`
        : '确定要交卷吗？',
      okText: '确定交卷',
      cancelText: '继续答题',
      onOk: () => {
        calculateResult()
      }
    })
  }

  // 计算结果
  const calculateResult = async () => {
    let correct = 0
    const detailedResults = []

    questions.forEach(q => {
      const userAnswer = answers[q.id]
      const isCorrect = userAnswer === q.correctAnswer
      if (isCorrect) correct++

      detailedResults.push({
        questionId: q.id,
        question: q.question,
        userAnswer,
        correctAnswer: q.correctAnswer,
        isCorrect,
        explanation: q.explanation
      })
    })

    const score = Math.round((correct / questions.length) * 100)
    const duration = examConfig.duration * 60 - timeLeft

    // 保存考试记录
    try {
      await api.post('/study/record', {
        type: 'mock_exam',
        content: {
          totalQuestions: questions.length,
          correctAnswers: correct,
          score,
          duration,
          detailedResults
        },
        duration,
        score,
        totalQuestions: questions.length,
        correctAnswers: correct
      })
    } catch (error) {
      console.error('保存考试记录失败:', error)
    }

    setResult({
      total: questions.length,
      correct,
      wrong: questions.length - correct,
      score,
      duration: Math.floor(duration / 60),
      detailedResults
    })

    setIsFinished(true)
    setIsStarted(false)
  }

  // 倒计时
  useEffect(() => {
    if (isStarted && timeLeft > 0) {
      const timer = setTimeout(() => {
        setTimeLeft(t => t - 1)
      }, 1000)
      return () => clearTimeout(timer)
    } else if (timeLeft === 0 && isStarted) {
      message.info('考试时间到，自动交卷')
      calculateResult()
    }
  }, [timeLeft, isStarted])

  // 格式化时间
  const formatTime = (seconds) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`
  }

  // 渲染配置界面
  if (showConfig) {
    return (
      <div style={{ maxWidth: 800, margin: '0 auto', padding: 24 }}>
        <Card title="模拟考试配置" style={{ marginBottom: 24 }}>
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <div>
              <h3>选择题型</h3>
              <Radio.Group
                value={examConfig?.type}
                onChange={e => setExamConfig(prev => ({ ...prev, type: e.target.value }))}
              >
                <Space wrap>
                  {questionTypes.map(type => (
                    <Radio.Button key={type.value} value={type.value}>
                      {type.label}
                    </Radio.Button>
                  ))}
                </Space>
              </Radio.Group>
            </div>

            <div>
              <h3>考试时长</h3>
              <Radio.Group
                value={examConfig?.duration}
                onChange={e => setExamConfig(prev => ({ ...prev, duration: e.target.value }))}
              >
                <Space>
                  {durationOptions.map(option => (
                    <Radio.Button key={option.value} value={option.value}>
                      {option.label}
                    </Radio.Button>
                  ))}
                </Space>
              </Radio.Group>
            </div>

            <div>
              <h3>题目数量</h3>
              <Radio.Group
                value={examConfig?.count}
                onChange={e => setExamConfig(prev => ({ ...prev, count: e.target.value }))}
              >
                <Space>
                  {countOptions.map(option => (
                    <Radio.Button key={option.value} value={option.value}>
                      {option.label}
                    </Radio.Button>
                  ))}
                </Space>
              </Radio.Group>
            </div>

            <Button
              type="primary"
              size="large"
              loading={loading}
              onClick={startExam}
              style={{ width: '100%' }}
            >
              开始考试
            </Button>
          </Space>
        </Card>

        <Card title="考试说明">
          <ul>
            <li>模拟考试将从题库中随机抽取题目</li>
            <li>考试结束前可以修改答案</li>
            <li>考试时间结束后系统将自动交卷</li>
            <li>交卷后可查看答案解析</li>
            <li>考试成绩将保存到学习记录中</li>
          </ul>
        </Card>
      </div>
    )
  }

  // 渲染考试结果
  if (isFinished && result) {
    return (
      <div style={{ maxWidth: 1000, margin: '0 auto', padding: 24 }}>
        <Result
          status={result.score >= 60 ? 'success' : 'error'}
          title={
            <div>
              <h2>考试完成！</h2>
              <h1 style={{ fontSize: 48, margin: '20px 0' }}>{result.score}分</h1>
            </div>
          }
          subTitle={`用时 ${result.duration} 分钟，正确率 ${Math.round((result.correct / result.total) * 100)}%`}
          extra={[
            <Button key="review" type="primary" onClick={() => setShowResultDetail(true)}>
              查看详情
            </Button>,
            <Button key="new" onClick={() => window.location.reload()}>
              再次考试
            </Button>,
            <Button key="home" onClick={() => navigate('/')}>
              返回首页
            </Button>
          ]}
        />

        <Row gutter={16} style={{ marginTop: 24 }}>
          <Col span={6}>
            <Card>
              <Statistic title="总题数" value={result.total} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="正确题数"
                value={result.correct}
                valueStyle={{ color: '#3f8600' }}
                prefix={<CheckCircleOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="错误题数"
                value={result.wrong}
                valueStyle={{ color: '#cf1322' }}
                prefix={<CloseCircleOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic title="用时" value={result.duration} suffix="分钟" />
            </Card>
          </Col>
        </Row>

        {/* 答题详情 */}
        <Modal
          title="答题详情"
          visible={showResultDetail}
          onCancel={() => setShowResultDetail(false)}
          footer={null}
          width={800}
        >
          {result.detailedResults.map((item, index) => (
            <Card key={index} style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h4>第 {index + 1} 题</h4>
                <Tag color={item.isCorrect ? 'success' : 'error'}>
                  {item.isCorrect ? '正确' : '错误'}
                </Tag>
              </div>
              <p>{item.question}</p>
              <p>您的答案：{item.userAnswer || '未作答'}</p>
              {!item.isCorrect && (
                <p>正确答案：{item.correctAnswer}</p>
              )}
              {item.explanation && (
                <p style={{ color: '#666' }}>解析：{item.explanation}</p>
              )}
            </Card>
          ))}
        </Modal>
      </div>
    )
  }

  // 渲染考试界面
  if (isStarted && questions.length > 0) {
    const currentQuestion = questions[currentQuestionIndex]
    const progress = ((currentQuestionIndex + 1) / questions.length) * 100
    const answeredCount = Object.keys(answers).length

    return (
      <div style={{ maxWidth: 800, margin: '0 auto', padding: 24 }}>
        {/* 考试信息栏 */}
        <Card style={{ marginBottom: 24 }}>
          <Row justify="space-between" align="middle">
            <Col>
              <Space>
                <span>第 {currentQuestionIndex + 1} 题 / 共 {questions.length} 题</span>
                <Divider type="vertical" />
                <span>已答 {answeredCount} 题</span>
              </Space>
            </Col>
            <Col>
              <Space>
                <ClockCircleOutlined style={{ color: timeLeft < 300 ? '#ff4d4f' : '#1890ff' }} />
                <span style={{ color: timeLeft < 300 ? '#ff4d4f' : '#1890ff' }}>
                  {formatTime(timeLeft)}
                </span>
              </Space>
            </Col>
          </Row>
          <Progress percent={progress} style={{ marginTop: 16 }} showInfo={false} />
        </Card>

        {/* 题目卡片 */}
        <Card title={currentQuestion.question} style={{ marginBottom: 24 }}>
          {currentQuestion.type === 'single_choice' && (
            <Radio.Group
              value={answers[currentQuestion.id]}
              onChange={e => handleAnswer(currentQuestion.id, e.target.value)}
            >
              <Space direction="vertical" style={{ width: '100%' }}>
                {currentQuestion.options.map((option, index) => (
                  <Radio key={index} value={index}>
                    {option}
                  </Radio>
                ))}
              </Space>
            </Radio.Group>
          )}

          {currentQuestion.type === 'multiple_choice' && (
            <Checkbox.Group
              value={answers[currentQuestion.id] || []}
              onChange={values => handleAnswer(currentQuestion.id, values)}
            >
              <Space direction="vertical" style={{ width: '100%' }}>
                {currentQuestion.options.map((option, index) => (
                  <Checkbox key={index} value={index}>
                    {option}
                  </Checkbox>
                ))}
              </Space>
            </Checkbox.Group>
          )}

          {currentQuestion.type === 'blank_filling' && (
            <Input
              value={answers[currentQuestion.id] || ''}
              onChange={e => handleAnswer(currentQuestion.id, e.target.value)}
              placeholder="请填写答案"
              size="large"
            />
          )}
        </Card>

        {/* 操作按钮 */}
        <Card>
          <Row justify="space-between">
            <Col>
              <Button
                disabled={currentQuestionIndex === 0}
                onClick={prevQuestion}
              >
                上一题
              </Button>
            </Col>
            <Col>
              <Space>
                <Button onClick={() => navigate('/')}>
                  退出考试
                </Button>
                {currentQuestionIndex < questions.length - 1 ? (
                  <Button type="primary" onClick={nextQuestion}>
                    下一题
                  </Button>
                ) : (
                  <Button type="primary" onClick={submitExam}>
                    交卷
                  </Button>
                )}
              </Space>
            </Col>
          </Row>
        </Card>

        {/* 题目导航 */}
        <Card title="题目导航" style={{ marginTop: 24 }}>
          <Row gutter={[8, 8]}>
            {questions.map((q, index) => {
              const isAnswered = answers[q.id] !== undefined
              const isCurrent = index === currentQuestionIndex
              return (
                <Col key={index}>
                  <Button
                    type={isCurrent ? 'primary' : 'default'}
                    style={{
                      backgroundColor: !isCurrent && isAnswered ? '#52c41a' : undefined,
                      borderColor: !isCurrent && isAnswered ? '#52c41a' : undefined
                    }}
                    onClick={() => setCurrentQuestionIndex(index)}
                  >
                    {index + 1}
                  </Button>
                </Col>
              )
            })}
          </Row>
        </Card>
      </div>
    )
  }

  return null
}

export default MockExam