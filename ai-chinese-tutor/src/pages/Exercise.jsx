import React, { useState, useEffect } from 'react'
import { Card, Tabs, Button, List, Progress, Badge, Avatar, Empty, message, Spin } from 'antd'
import { BookOutlined, FileTextOutlined, BarChartOutlined, UserOutlined, PlusOutlined, CheckCircleOutlined, ClockCircleOutlined, ReloadOutlined, TrophyOutlined } from '@ant-design/icons'
import { studyRecordService } from '../services/studyRecordService'
import { questionService } from '../services/questionService'

const { TabPane } = Tabs

const Exercise = () => {
  // 状态管理
  const [currentTab, setCurrentTab] = useState('1')
  const [exercises, setExercises] = useState([])
  const [completedExercises, setCompletedExercises] = useState([])
  const [recommendedExercises, setRecommendedExercises] = useState([])
  const [loading, setLoading] = useState(false)
  const [currentExercise, setCurrentExercise] = useState(null)
  const [isExerciseStarted, setIsExerciseStarted] = useState(false)
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
  const [answers, setAnswers] = useState([])
  const [exerciseResult, setExerciseResult] = useState(null)
  const [studyStats, setStudyStats] = useState({
    totalExercises: 0,
    completedExercises: 0,
    averageScore: 0,
    streakDays: 0
  })
  
  // 初始化数据
  useEffect(() => {
    loadExercises()
    loadCompletedExercises()
    loadRecommendedExercises()
    loadStudyStats()
  }, [])
  
  // 加载练习列表
  const loadExercises = async () => {
    try {
      setLoading(true)
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 500))
      
      // 模拟练习数据
      const mockExercises = [
        {
          id: 1,
          title: '小学语文基础知识练习',
          type: '基础知识',
          difficulty: '简单',
          duration: '15分钟',
          questionCount: 20,
          completionRate: 35,
          lastStudyTime: '2023-06-15'
        },
        {
          id: 2,
          title: '成语填空专项训练',
          type: '成语',
          difficulty: '中等',
          duration: '20分钟',
          questionCount: 25,
          completionRate: 0,
          lastStudyTime: null
        },
        {
          id: 3,
          title: '阅读理解提升练习',
          type: '阅读理解',
          difficulty: '较难',
          duration: '30分钟',
          questionCount: 15,
          completionRate: 60,
          lastStudyTime: '2023-06-10'
        },
        {
          id: 4,
          title: '古诗词默写与理解',
          type: '古诗词',
          difficulty: '中等',
          duration: '25分钟',
          questionCount: 20,
          completionRate: 100,
          lastStudyTime: '2023-06-12'
        },
        {
          id: 5,
          title: '病句修改练习',
          type: '语法',
          difficulty: '简单',
          duration: '15分钟',
          questionCount: 15,
          completionRate: 80,
          lastStudyTime: '2023-06-08'
        }
      ]
      
      setExercises(mockExercises)
    } catch (error) {
      message.error('加载练习列表失败')
    } finally {
      setLoading(false)
    }
  }
  
  // 加载已完成的练习
  const loadCompletedExercises = async () => {
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 300))
      
      // 模拟已完成练习数据
      const mockCompletedExercises = [
        {
          id: 4,
          title: '古诗词默写与理解',
          type: '古诗词',
          completionDate: '2023-06-12',
          score: 92,
          rank: '优秀',
          duration: '22分钟'
        },
        {
          id: 5,
          title: '病句修改练习',
          type: '语法',
          completionDate: '2023-06-08',
          score: 85,
          rank: '良好',
          duration: '13分钟'
        }
      ]
      
      setCompletedExercises(mockCompletedExercises)
    } catch (error) {
      message.error('加载已完成练习失败')
    }
  }
  
  // 加载推荐练习
  const loadRecommendedExercises = async () => {
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 300))
      
      // 模拟推荐练习数据
      const mockRecommendedExercises = [
        {
          id: 6,
          title: '常见错别字辨析',
          type: '字形',
          reason: '根据你的学习记录，建议加强字形辨析练习',
          priority: 'high'
        },
        {
          id: 7,
          title: '标点符号使用规范',
          type: '标点',
          reason: '标点符号使用错误率较高，建议专项练习',
          priority: 'medium'
        },
        {
          id: 8,
          title: '修辞手法识别',
          type: '修辞',
          reason: '修辞手法识别是考试重点，建议多加练习',
          priority: 'medium'
        }
      ]
      
      setRecommendedExercises(mockRecommendedExercises)
    } catch (error) {
      message.error('加载推荐练习失败')
    }
  }
  
  // 加载学习统计数据
  const loadStudyStats = async () => {
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 200))
      
      // 模拟学习统计数据
      const mockStudyStats = {
        totalExercises: 12,
        completedExercises: 5,
        averageScore: 85,
        streakDays: 3
      }
      
      setStudyStats(mockStudyStats)
    } catch (error) {
      message.error('加载学习统计失败')
    }
  }
  
  // 开始练习
  const startExercise = async (exercise) => {
    try {
      setLoading(true)
      message.loading({ content: '正在获取题目...', key: 'loading' })

      // 从后端获取真实题目
      const questions = await questionService.getRandomQuestions({
        type: exercise.type,
        count: exercise.questionCount || 10,
        difficulty: exercise.difficulty,
        year: exercise.year
      })

      if (questions.length === 0) {
        message.warning({ content: '暂无相关题目，使用模拟题目', key: 'loading' })
        // 如果没有真实题目，使用模拟题目
        const mockQuestions = generateMockQuestions(exercise.questionCount || 10)

        setCurrentExercise({
          ...exercise,
          questions: mockQuestions
        })
        setAnswers(new Array(mockQuestions.length).fill(null))
      } else {
        message.success({ content: `成功获取 ${questions.length} 道题目`, key: 'loading' })
        setCurrentExercise({
          ...exercise,
          questions: questions
        })
        setAnswers(new Array(questions.length).fill(null))
      }

      setCurrentQuestionIndex(0)
      setIsExerciseStarted(true)
      setExerciseResult(null)
    } catch (error) {
      message.error({ content: '获取题目失败，使用模拟题目', key: 'loading' })
      console.error('获取题目失败:', error)

      // 出错时使用模拟题目
      const mockQuestions = generateMockQuestions(exercise.questionCount || 10)

      setCurrentExercise({
        ...exercise,
        questions: mockQuestions
      })
      setAnswers(new Array(mockQuestions.length).fill(null))
      setCurrentQuestionIndex(0)
      setIsExerciseStarted(true)
      setExerciseResult(null)
    } finally {
      setLoading(false)
    }
  }
  
  // 生成模拟题目
  const generateMockQuestions = (count) => {
    const questionTypes = [
      {
        type: 'single_choice',
        formatQuestion: () => ({
          id: Math.floor(Math.random() * 1000),
          question: '下列哪个成语使用正确？',
          options: [
            '他的文章错别字连篇，真是文不加点。',
            '小明夸夸其谈，终于说服了老师。',
            '这部电影情节跌宕起伏，扣人心弦。',
            '他上课总是心不在焉，全神贯注。'
          ],
          correctAnswer: 2
        })
      },
      {
        type: 'multiple_choice',
        formatQuestion: () => ({
          id: Math.floor(Math.random() * 1000),
          question: '下列句子中有错别字的是哪些？',
          options: [
            '我们要发扬勤俭节约的传统美德。',
            '他的学习成绩一直名列前矛。',
            '这篇文章写得深入浅出，通俗易懂。',
            '同学们兴高彩烈地参加了运动会。'
          ],
          correctAnswer: [1, 3]
        })
      },
      {
        type: 'blank_filling',
        formatQuestion: () => ({
          id: Math.floor(Math.random() * 1000),
          question: '补全诗句：床前明月光，疑是地上霜。举头望明月，__。',
          correctAnswer: '低头思故乡'
        })
      },
      {
        type: 'true_false',
        formatQuestion: () => ({
          id: Math.floor(Math.random() * 1000),
          question: '"三个臭皮匠，顶个诸葛亮"是一个褒义的成语。',
          correctAnswer: true
        })
      }
    ]
    
    const questions = []
    for (let i = 0; i < count; i++) {
      const randomTypeIndex = Math.floor(Math.random() * questionTypes.length)
      const questionType = questionTypes[randomTypeIndex]
      questions.push(questionType.formatQuestion())
    }
    
    return questions
  }
  
  // 选择答案
  const selectAnswer = (answer) => {
    const newAnswers = [...answers]
    newAnswers[currentQuestionIndex] = answer
    setAnswers(newAnswers)
  }

  // 检查当前答案是否正确
  const checkCurrentAnswer = () => {
    const question = currentExercise.questions[currentQuestionIndex]
    const currentAnswer = answers[currentQuestionIndex]
    
    if (currentAnswer === null) return false
    
    if (Array.isArray(question.correctAnswer)) {
      // 多选题
      return JSON.stringify(currentAnswer.sort()) === JSON.stringify(question.correctAnswer.sort())
    } else {
      // 单选题、判断题、填空题
      // 对于填空题，做一些模糊匹配处理
      if (question.type === 'blank_filling') {
        return currentAnswer.trim() === question.correctAnswer.trim()
      }
      return currentAnswer === question.correctAnswer
    }
  }
  
  // 下一题
  const nextQuestion = () => {
    if (currentQuestionIndex < currentExercise.questions.length - 1) {
      setCurrentQuestionIndex(prev => prev + 1)
    } else {
      // 完成练习，计算结果
      calculateResult()
    }
  }
  
  // 上一题
  const prevQuestion = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(prev => prev - 1)
    }
  }
  
  // 计算练习结果
  const calculateResult = () => {
    if (!currentExercise) return
    
    let correctCount = 0
    answers.forEach((answer, index) => {
      const question = currentExercise.questions[index]
      let isCorrect = false
      
      if (Array.isArray(question.correctAnswer)) {
        // 多选题
        isCorrect = JSON.stringify(answer.sort()) === JSON.stringify(question.correctAnswer.sort())
      } else {
        // 单选题、判断题、填空题
        isCorrect = answer === question.correctAnswer
      }
      
      if (isCorrect) {
        correctCount++
      }
    })
    
    const score = Math.round((correctCount / answers.length) * 100)
    let rank = '不及格'
    if (score >= 90) rank = '优秀'
    else if (score >= 80) rank = '良好'
    else if (score >= 60) rank = '及格'
    
    const result = {
      score,
      rank,
      correctCount,
      totalCount: answers.length,
      startTime: new Date().getTime() - (currentExercise.questions.length * 30 * 1000), // 模拟用时
      endTime: new Date().getTime()
    }
    
    setExerciseResult(result)
    setIsExerciseStarted(false)
    
    // 保存练习结果到数据库
    saveExerciseResult(result).then(() => {
      // 刷新已完成练习列表和学习统计
      loadCompletedExercises()
      loadStudyStats()
      message.success('练习完成！成绩已保存')
    }).catch(error => {
      console.error('保存练习结果失败:', error)
      message.error('保存练习结果失败')
    })
  }
  
    
  // 重新开始练习
  const restartExercise = () => {
    if (currentExercise) {
      setCurrentQuestionIndex(0)
      setAnswers(new Array(currentExercise.questions.length).fill(null))
      setIsExerciseStarted(true)
      setExerciseResult(null)
    }
  }
  
  // 返回练习列表
  const returnToList = () => {
    setCurrentExercise(null)
    setIsExerciseStarted(false)
    setExerciseResult(null)
  }
  
  // 渲染练习列表项
  const renderExerciseItem = (exercise) => (
    <Card key={exercise.id} className="exercise-item-card">
      <div className="exercise-item-header">
        <h3>{exercise.title}</h3>
        <Badge color={getDifficultyColor(exercise.difficulty)} text={exercise.difficulty} />
      </div>
      <div className="exercise-item-info">
        <div className="info-item">
          <span className="info-label">类型：</span>
          <span className="info-value">{exercise.type}</span>
        </div>
        <div className="info-item">
          <span className="info-label">时长：</span>
          <span className="info-value">{exercise.duration}</span>
        </div>
        <div className="info-item">
          <span className="info-label">题目数量：</span>
          <span className="info-value">{exercise.questionCount}题</span>
        </div>
        {exercise.lastStudyTime && (
          <div className="info-item">
            <span className="info-label">上次学习：</span>
            <span className="info-value">{exercise.lastStudyTime}</span>
          </div>
        )}
      </div>
      {exercise.completionRate > 0 && (
        <div className="exercise-item-progress">
          <div className="progress-header">
            <span>完成进度</span>
            <span>{exercise.completionRate}%</span>
          </div>
          <Progress percent={exercise.completionRate} status={exercise.completionRate === 100 ? "success" : "active"} />
        </div>
      )}
      <div className="exercise-item-actions">
        <Button 
          type="primary" 
          onClick={() => startExercise(exercise)}
          className="start-btn"
        >
          {exercise.completionRate === 100 ? '重新练习' : '继续练习'}
        </Button>
      </div>
    </Card>
  )
  
  // 根据难度获取颜色
  const getDifficultyColor = (difficulty) => {
    const colorMap = {
      '简单': 'green',
      '中等': 'orange',
      '较难': 'red',
      '困难': 'magenta'
    }
    return colorMap[difficulty] || 'blue'
  }
  
  // 渲染当前问题
  const renderCurrentQuestion = () => {
    if (!currentExercise || !currentExercise.questions[currentQuestionIndex]) {
      return <Empty description="暂无题目" />
    }
    
    const question = currentExercise.questions[currentQuestionIndex]
    const currentAnswer = answers[currentQuestionIndex]
    const isAnswerCorrect = currentAnswer !== null ? checkCurrentAnswer() : null
    
    return (
      <Card className="exercise-question-card">
        <div className="question-header">
          <h3>题目 {currentQuestionIndex + 1}/{currentExercise.questions.length}</h3>
          <Badge color={getDifficultyColor(currentExercise.difficulty)} text={currentExercise.difficulty} />
        </div>
        <div className="question-content">
          {/* 对于填空题，题目文本会在填空部分显示，这里不重复显示 */}
          {question.type !== 'blank_filling' && (
            <p className="question-text">{question.question}</p>
          )}
          
          {question.options && (
            <div className="question-options">
              {question.options.map((option, index) => {
                // 检查当前选项是否正确
                const isOptionCorrect = Array.isArray(question.correctAnswer) 
                  ? question.correctAnswer.includes(index)
                  : question.correctAnswer === index;
                
                // 判断是否需要显示正确/错误图标
                const showIcon = currentAnswer !== null;
                
                return (
                  <Button
                    key={index}
                    type={currentAnswer === index ? 'primary' : 'default'}
                    className={`option-btn ${currentAnswer === index ? 'selected' : ''} ${showIcon && isOptionCorrect ? 'correct-option' : ''} ${showIcon && currentAnswer === index && !isOptionCorrect ? 'incorrect-option' : ''}`}
                    onClick={() => selectAnswer(index)}
                    block
                    disabled={showIcon}
                    icon={showIcon && (
                      isOptionCorrect ? 
                        <CheckCircleOutlined style={{ color: '#52c41a' }} /> : 
                        (currentAnswer === index ? 
                          <CheckCircleOutlined style={{ color: '#ff4d4f' }} /> : 
                          null
                        )
                    )}
                  >
                    {String.fromCharCode(65 + index)}. {option}
                  </Button>
                )
              })}
            </div>
          )}
          
          {!question.options && question.type === 'blank_filling' && (
            <div className="blank-filling-container" style={{ marginTop: '20px' }}>
              <p style={{ color: '#666', marginBottom: '10px' }}>请在下方输入答案：</p>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                marginBottom: '20px'
              }}>
                <span style={{ fontSize: '18px' }}>
                  {question.question.replace(/__+./, (match) => {
                    // 找到空白部分（__。）并替换为输入框
                    return ''
                  })}
                </span>
                <input
                  type="text"
                  className="blank-filling-input"
                  value={currentAnswer || ''}
                  onChange={(e) => selectAnswer(e.target.value)}
                  placeholder="请输入答案"
                  style={{
                    width: '200px',
                    padding: '8px 12px',
                    fontSize: '16px',
                    border: `2px solid ${isAnswerCorrect !== null ? (isAnswerCorrect ? '#52c41a' : '#ff4d4f') : '#d9d9d9'}`,
                    borderRadius: '6px',
                    outline: 'none',
                    transition: 'border-color 0.3s',
                    boxSizing: 'border-box',
                    display: 'inline-block'
                  }}
                  onFocus={(e) => e.target.style.borderColor = isAnswerCorrect !== null ? (isAnswerCorrect ? '#52c41a' : '#ff4d4f') : '#40a9ff'}
                  onBlur={(e) => e.target.style.borderColor = isAnswerCorrect !== null ? (isAnswerCorrect ? '#52c41a' : '#ff4d4f') : '#d9d9d9'}
                  disabled={isAnswerCorrect !== null}
                />
                <span style={{ fontSize: '18px' }}>
                  {question.question.replace(/.*__+/, '')}
                </span>
              </div>
              {isAnswerCorrect !== null && (
                <div style={{
                  marginTop: '10px',
                  color: isAnswerCorrect ? '#52c41a' : '#ff4d4f',
                  fontSize: '16px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px'
                }}>
                  {isAnswerCorrect ?
                    <><CheckCircleOutlined /> 回答正确！</> :
                    <><CheckCircleOutlined /> 回答错误。正确答案是：{question.correctAnswer}</>
                  }
                </div>
              )}
              <div style={{
                marginTop: '5px',
                color: '#999',
                fontSize: '14px'
              }}>
                提示：输入答案后点击"下一题"继续
              </div>
            </div>
          )}
          
          {!question.options && question.type === 'true_false' && (
            <div className="true-false-options">
              <Button
                type={currentAnswer === true ? 'primary' : 'default'}
                onClick={() => selectAnswer(true)}
                className={`true-false-btn ${isAnswerCorrect !== null ? (question.correctAnswer === true ? 'correct-option' : (currentAnswer === true ? 'incorrect-option' : '')) : ''}`}
                disabled={isAnswerCorrect !== null}
                icon={isAnswerCorrect !== null && question.correctAnswer === true && (
                  <CheckCircleOutlined style={{ color: '#52c41a' }} />
                )}
              >
                正确
              </Button>
              <Button
                type={currentAnswer === false ? 'primary' : 'default'}
                onClick={() => selectAnswer(false)}
                className={`true-false-btn ${isAnswerCorrect !== null ? (question.correctAnswer === false ? 'correct-option' : (currentAnswer === false ? 'incorrect-option' : '')) : ''}`}
                disabled={isAnswerCorrect !== null}
                icon={isAnswerCorrect !== null && question.correctAnswer === false && (
                  <CheckCircleOutlined style={{ color: '#52c41a' }} />
                )}
              >
                错误
              </Button>
              {isAnswerCorrect !== null && (
                <div style={{ 
                  marginTop: '10px', 
                  color: isAnswerCorrect ? '#52c41a' : '#ff4d4f', 
                  fontSize: '16px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px'
                }}>
                  {isAnswerCorrect ? 
                    <CheckCircleOutlined /> : 
                    <CheckCircleOutlined />
                  }
                  {isAnswerCorrect ? 
                    '回答正确！' : 
                    `回答错误。正确答案是：${question.correctAnswer ? '正确' : '错误'}`
                  }
                </div>
              )}
            </div>
          )}
        </div>
        <div className="question-actions">
          <Button 
            onClick={prevQuestion} 
            disabled={currentQuestionIndex === 0}
            className="prev-btn"
          >
            上一题
          </Button>
          <Button 
            type="primary" 
            onClick={nextQuestion} 
            disabled={currentAnswer === null}
            className="next-btn"
          >
            {currentQuestionIndex < currentExercise.questions.length - 1 ? '下一题' : '完成'}
          </Button>
        </div>
      </Card>
    )
  }
  
  // 渲染练习结果
  const renderExerciseResult = () => {
    if (!exerciseResult) {
      return <Empty description="暂无练习结果" />
    }
    
    return (
      <Card className="exercise-result-card">
        <div className="result-header">
          <TrophyOutlined className="result-icon" />
          <h2>练习完成！</h2>
        </div>
        <div className="result-stats">
          <div className="stat-item score">
            <div className="stat-value">{exerciseResult.score}</div>
            <div className="stat-label">得分</div>
          </div>
          <div className="stat-item rank">
            <div className="stat-value">{exerciseResult.rank}</div>
            <div className="stat-label">等级</div>
          </div>
          <div className="stat-item correct">
            <div className="stat-value">{exerciseResult.correctCount}</div>
            <div className="stat-label">正确题数</div>
          </div>
          <div className="stat-item total">
            <div className="stat-value">{exerciseResult.totalCount}</div>
            <div className="stat-label">总题数</div>
          </div>
        </div>
        <div className="result-progress">
          <Progress 
            percent={exerciseResult.score} 
            status={exerciseResult.score >= 60 ? "success" : "exception"} 
            strokeWidth={20} 
            showInfo={false}
          />
        </div>
        <div className="result-actions">
          <Button 
            type="primary" 
            onClick={restartExercise}
            className="restart-btn"
          >
            重新练习
          </Button>
          <Button 
            onClick={returnToList}
            className="return-btn"
          >
            返回列表
          </Button>
        </div>
      </Card>
    )
  }
  
  // 保存练习结果到数据库
  const saveExerciseResult = async (result) => {
    try {
      // 计算练习时长（秒）
      const duration = Math.round((result.endTime - result.startTime) / 1000);

      // 准备学习记录数据
      const recordData = {
        type: 'exercise',
        content: {
          title: currentExercise.title,
          type: currentExercise.type || '综合练习',
          difficulty: currentExercise.difficulty || '中等',
          questionCount: result.totalCount,
          correctCount: result.correctCount,
          score: result.score
        },
        duration: duration,
        score: result.score,
        totalQuestions: result.totalCount,
        correctAnswers: result.correctCount
      };

      // 调用服务保存记录
      const saveResult = await studyRecordService.addStudyRecord(recordData);

      if (saveResult.success) {
        console.log('练习记录保存成功');
      } else {
        console.error('练习记录保存失败:', saveResult.message);
      }
    } catch (error) {
      console.error('保存练习记录出错:', error);
      message.error('保存练习记录失败');
    }
  };

  // 渲染学习统计卡片
  const renderStudyStatsCard = () => (
    <Card className="study-stats-card">
      <h3>学习统计</h3>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-number">{studyStats.totalExercises}</div>
          <div className="stat-name">总练习数</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{studyStats.completedExercises}</div>
          <div className="stat-name">已完成</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{studyStats.averageScore}%</div>
          <div className="stat-name">平均分</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">{studyStats.streakDays}</div>
          <div className="stat-name">连续学习(天)</div>
        </div>
      </div>
    </Card>
  )
  
  // 渲染已完成练习项
  const renderCompletedExerciseItem = (exercise) => (
    <List.Item key={exercise.id} className="completed-exercise-item">
      <List.Item.Meta
        avatar={<Avatar>{exercise.type[0]}</Avatar>}
        title={exercise.title}
        description={
          <>
            <div className="completed-info">
              <span className="info-item">完成日期：{exercise.completionDate}</span>
              <span className={`info-item score ${exercise.score >= 90 ? 'excellent' : exercise.score >= 80 ? 'good' : exercise.score >= 60 ? 'pass' : 'fail'}`}>
                得分：{exercise.score}
              </span>
              <span className="info-item">等级：{exercise.rank}</span>
              <span className="info-item">用时：{exercise.duration}</span>
            </div>
          </>
        }
      />
      <Button 
        type="text" 
        onClick={() => startExercise({id: exercise.id, title: exercise.title, type: exercise.type})}
      >
        再练一次
      </Button>
    </List.Item>
  )
  
  // 渲染推荐练习项
  const renderRecommendedExerciseItem = (exercise) => (
    <Card key={exercise.id} className={`recommended-exercise-item priority-${exercise.priority}`}>
      <div className="recommended-header">
        <h4>{exercise.title}</h4>
        <Badge color={exercise.priority === 'high' ? 'red' : 'orange'} text={exercise.priority === 'high' ? '高' : '中'} />
      </div>
      <p className="recommended-reason">{exercise.reason}</p>
      <Button 
        type="primary" 
        onClick={() => startExercise({id: exercise.id, title: exercise.title, type: exercise.type})}
        size="small"
      >
        立即练习
      </Button>
    </Card>
  )
  
  // 主内容渲染
  const renderMainContent = () => {
    // 如果正在进行练习或已完成练习显示结果
    if (currentExercise) {
      if (isExerciseStarted) {
        return renderCurrentQuestion()
      } else if (exerciseResult) {
        return renderExerciseResult()
      }
    }
    
    // 否则显示练习列表和统计
    return (
      <>
        {renderStudyStatsCard()}
        
        <Tabs activeKey={currentTab} onChange={setCurrentTab} className="exercise-tabs">
          <TabPane tab="全部练习" key="1">
            {loading ? (
              <div className="loading-wrapper">
                <Spin size="large" tip="加载中..." />
              </div>
            ) : exercises.length > 0 ? (
              <div className="exercises-grid">
                {exercises.map(renderExerciseItem)}
              </div>
            ) : (
              <Empty description="暂无练习" />
            )}
          </TabPane>
          
          <TabPane tab="已完成" key="2">
            {completedExercises.length > 0 ? (
              <List
                dataSource={completedExercises}
                renderItem={renderCompletedExerciseItem}
              />
            ) : (
              <Empty description="暂无已完成的练习" />
            )}
          </TabPane>
          
          <TabPane tab="推荐练习" key="3">
            {recommendedExercises.length > 0 ? (
              <div className="recommended-exercises-list">
                {recommendedExercises.map(renderRecommendedExerciseItem)}
              </div>
            ) : (
              <Empty description="暂无推荐练习" />
            )}
          </TabPane>
        </Tabs>
      </>
    )
  }
  
  return (
    <div className="exercise-container">
      {renderMainContent()}
    </div>
  )
}

export default Exercise