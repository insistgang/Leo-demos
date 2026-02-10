import { useState, useEffect } from 'react'
import { Card, Tabs, Input, Button, List, Badge, Tag, Empty, Spin, message, Radio, Space, Checkbox } from 'antd'
import { SearchOutlined, BookOutlined, HistoryOutlined, StarOutlined, StarFilled, CheckCircleOutlined, CloseCircleOutlined }
  from '@ant-design/icons'
import { questionService } from '../services/questionService'
import { studyRecordService } from '../services/studyRecordService'

const { TabPane } = Tabs
const { Search } = Input

const Idiom = () => {
  // 状态管理
  const [idiomQuestions, setIdiomQuestions] = useState([])
  const [currentTab, setCurrentTab] = useState('1')
  const [selectedIdiom, setSelectedIdiom] = useState(null)
  const [searchText, setSearchText] = useState('')
  const [filteredQuestions, setFilteredQuestions] = useState([])
  const [loading, setLoading] = useState(false)
  const [favorites, setFavorites] = useState(new Set())
  const [searchHistory, setSearchHistory] = useState([])
  const [exerciseMode, setExerciseMode] = useState(false)
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
  const [selectedAnswer, setSelectedAnswer] = useState(null)
  const [exerciseResult, setExerciseResult] = useState(null)
  const [showExplanation, setShowExplanation] = useState(false)
  const [answeredQuestions, setAnsweredQuestions] = useState([])
  const [correctCount, setCorrectCount] = useState(0)

  // 组件挂载时加载数据
  useEffect(() => {
    loadIdiomQuestions()
    loadFavorites()
    loadSearchHistory()
  }, [])

  // 过滤成语题目列表
  useEffect(() => {
    filterQuestions()
  }, [idiomQuestions, searchText])

  // 加载成语题目
  const loadIdiomQuestions = async () => {
    try {
      setLoading(true)
      const questions = await questionService.getQuestionsByType('idiom', { count: 20 })
      setIdiomQuestions(questions)
    } catch (error) {
      console.error('加载成语题目失败:', error)
      message.error('加载成语题目失败')
    } finally {
      setLoading(false)
    }
  }

  // 加载收藏的成语
  const loadFavorites = () => {
    const savedFavorites = localStorage.getItem('idiomFavorites')
    if (savedFavorites) {
      try {
        const favoritesSet = new Set(JSON.parse(savedFavorites))
        setFavorites(favoritesSet)
      } catch (error) {
        console.error('加载收藏数据失败', error)
      }
    }
  }

  // 加载搜索历史
  const loadSearchHistory = () => {
    const savedHistory = localStorage.getItem('idiomSearchHistory')
    if (savedHistory) {
      try {
        setSearchHistory(JSON.parse(savedHistory))
      } catch (error) {
        console.error('加载搜索历史失败', error)
      }
    }
  }

  // 保存收藏的成语
  const saveFavorites = (favoritesSet) => {
    try {
      localStorage.setItem('idiomFavorites', JSON.stringify(Array.from(favoritesSet)))
    } catch (error) {
      console.error('保存收藏数据失败', error)
    }
  }

  // 保存搜索历史
  const saveSearchHistory = (history) => {
    try {
      localStorage.setItem('idiomSearchHistory', JSON.stringify(history))
    } catch (error) {
      console.error('保存搜索历史失败', error)
    }
  }

  // 过滤题目
  const filterQuestions = () => {
    let filtered = [...idiomQuestions]

    // 搜索过滤
    if (searchText.trim()) {
      const searchLower = searchText.toLowerCase()
      filtered = filtered.filter(q =>
        q.question.toLowerCase().includes(searchLower) ||
        q.options.some(opt => opt.toLowerCase().includes(searchLower)) ||
        q.explanation.toLowerCase().includes(searchLower)
      )
    }

    setFilteredQuestions(filtered)
  }

  // 搜索成语
  const handleSearch = (value) => {
    setSearchText(value)

    // 添加到搜索历史
    if (value.trim()) {
      const newHistory = [value.trim(), ...searchHistory.filter(item => item !== value.trim())].slice(0, 10)
      setSearchHistory(newHistory)
      saveSearchHistory(newHistory)
    }
  }

  // 清除搜索历史
  const clearSearchHistory = () => {
    setSearchHistory([])
    saveSearchHistory([])
  }

  // 点击搜索历史项
  const handleHistoryClick = (item) => {
    setSearchText(item)
    filterQuestions()
  }

  // 查看题目详情
  const viewQuestionDetail = (question) => {
    setSelectedIdiom(question)
    setCurrentTab('2') // 切换到详情标签页
  }

  // 切换收藏状态
  const toggleFavorite = (questionId) => {
    const newFavorites = new Set(favorites)
    if (newFavorites.has(questionId)) {
      newFavorites.delete(questionId)
      message.success('已取消收藏')
    } else {
      newFavorites.add(questionId)
      message.success('已添加到收藏')
    }
    setFavorites(newFavorites)
    saveFavorites(newFavorites)
  }

  // 开始练习
  const startExercise = () => {
    // 随机打乱题目顺序
    const shuffled = [...idiomQuestions].sort(() => Math.random() - 0.5)
    setIdiomQuestions(shuffled)
    setCurrentQuestionIndex(0)
    setSelectedAnswer(null)
    setExerciseResult(null)
    setShowExplanation(false)
    setAnsweredQuestions([])
    setCorrectCount(0)
    setExerciseMode(true)
    setCurrentTab('5') // 切换到练习标签页
  }

  // 提交练习答案
  const submitAnswer = async () => {
    const currentQuestion = idiomQuestions[currentQuestionIndex]
    if (!currentQuestion || selectedAnswer === null) return

    let isCorrect = false
    if (currentQuestion.type === 'single_choice') {
      isCorrect = selectedAnswer === currentQuestion.correctAnswer
    } else if (currentQuestion.type === 'multiple_choice') {
      // 对于多选题，需要比较数组
      const correctArray = Array.isArray(currentQuestion.correctAnswer)
        ? currentQuestion.correctAnswer.sort()
        : [currentQuestion.correctAnswer]
      const selectedArray = Array.isArray(selectedAnswer)
        ? selectedAnswer.sort()
        : [selectedAnswer]
      isCorrect = JSON.stringify(correctArray) === JSON.stringify(selectedArray)
    }

    const score = isCorrect ? 10 : 0

    // 保存答题记录
    try {
      const recordData = {
        type: 'idiom',
        questionId: currentQuestion.id,
        score: score,
        totalQuestions: 1,
        correctAnswers: isCorrect ? 1 : 0,
        duration: 30, // 假设每题30秒
        content: {
          question: currentQuestion.question,
          userAnswer: currentQuestion.type === 'multiple_choice'
            ? (Array.isArray(selectedAnswer) ? selectedAnswer.join(', ') : selectedAnswer)
            : currentQuestion.options[selectedAnswer],
          correctAnswer: currentQuestion.type === 'multiple_choice'
            ? (Array.isArray(currentQuestion.correctAnswer)
              ? currentQuestion.correctAnswer.map(i => currentQuestion.options[i]).join(', ')
              : currentQuestion.options[currentQuestion.correctAnswer])
            : currentQuestion.options[currentQuestion.correctAnswer],
          type: currentQuestion.type
        }
      }

      const saveResult = await studyRecordService.addStudyRecord(recordData)
      if (!saveResult.success) {
        console.error('答题记录保存失败:', saveResult.message)
      }
    } catch (error) {
      console.error('保存答题记录出错:', error)
    }

    // 更新练习结果
    setExerciseResult({
      isCorrect,
      correctAnswer: currentQuestion.correctAnswer,
      explanation: currentQuestion.explanation
    })
    setShowExplanation(true)

    // 记录已答题
    setAnsweredQuestions([...answeredQuestions, currentQuestionIndex])
    if (isCorrect) {
      setCorrectCount(correctCount + 1)
    }
  }

  // 下一题
  const nextQuestion = () => {
    if (currentQuestionIndex < idiomQuestions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1)
      setSelectedAnswer(null)
      setExerciseResult(null)
      setShowExplanation(false)
    } else {
      // 练习结束
      finishExercise()
    }
  }

  // 结束练习
  const finishExercise = async () => {
    // 保存练习总结
    try {
      const recordData = {
        type: 'idiom',
        score: Math.round((correctCount / answeredQuestions.length) * 100),
        totalQuestions: answeredQuestions.length,
        correctAnswers: correctCount,
        duration: answeredQuestions.length * 30, // 估算总时长
        content: {
          title: '成语练习总结',
          type: 'exercise_summary',
          accuracy: Math.round((correctCount / answeredQuestions.length) * 100)
        }
      }

      const saveResult = await studyRecordService.addStudyRecord(recordData)
      if (saveResult.success) {
        console.log('成语练习总结保存成功')
      } else {
        console.error('成语练习总结保存失败:', saveResult.message)
      }
    } catch (error) {
      console.error('保存成语练习总结出错:', error)
    }

    message.success(`练习完成！正确率：${Math.round((correctCount / answeredQuestions.length) * 100)}%`)
    setExerciseMode(false)
    setCurrentTab('1')
  }

  // 渲染题目列表项
  const renderQuestionItem = (question) => (
    <Card
      key={question.id}
      className="idiom-item-card"
      hoverable
      onClick={() => viewQuestionDetail(question)}
      extra={
        <Button
          type="text"
          icon={favorites.has(question.id) ? <StarFilled style={{ color: '#ffd700' }} /> : <StarOutlined />}
          onClick={(e) => {
            e.stopPropagation()
            toggleFavorite(question.id)
          }}
        />
      }
    >
      <div className="idiom-item-header">
        <h3>{question.question}</h3>
        <p className="question-source">{question.source}</p>
      </div>
      <p className="explanation-preview">{question.explanation.substring(0, 50)}...</p>
      <div className="idiom-item-tags">
        <Tag color={question.difficulty === 'easy' ? 'green' : question.difficulty === 'hard' ? 'red' : 'orange'}>
          {question.difficulty === 'easy' ? '简单' : question.difficulty === 'hard' ? '困难' : '中等'}
        </Tag>
        <Tag color="purple">{question.year}年</Tag>
      </div>
    </Card>
  )

  // 渲染题目详情
  const renderQuestionDetail = () => {
    if (!selectedIdiom) {
      return <Empty description="请选择一个题目查看详情" />
    }

    return (
      <Card className="idiom-detail-card">
        <div className="idiom-detail-header">
          <h2>题目详情</h2>
          <div className="question-content">{selectedIdiom.question}</div>
          <div className="idiom-detail-tags">
            <Tag color={selectedIdiom.difficulty === 'easy' ? 'green' : selectedIdiom.difficulty === 'hard' ? 'red' : 'orange'}>
              {selectedIdiom.difficulty === 'easy' ? '简单' : selectedIdiom.difficulty === 'hard' ? '困难' : '中等'}
            </Tag>
            <Tag color="purple">{selectedIdiom.source}</Tag>
          </div>
        </div>

        <div className="idiom-detail-content">
          <div className="detail-section">
            <h4>选项</h4>
            <div className="options-list">
              {selectedIdiom.options.map((option, index) => (
                <div key={index} className="option-item">
                  <Tag color={index === selectedIdiom.correctAnswer ? 'green' : 'default'}>
                    {String.fromCharCode(65 + index)}. {option}
                  </Tag>
                  {index === selectedIdiom.correctAnswer && <CheckCircleOutlined style={{ color: 'green', marginLeft: 8 }} />}
                </div>
              ))}
            </div>
          </div>

          <div className="detail-section">
            <h4>正确答案</h4>
            <p>{selectedIdiom.options[selectedIdiom.correctAnswer]}</p>
          </div>

          <div className="detail-section">
            <h4>解析</h4>
            <p>{selectedIdiom.explanation}</p>
          </div>
        </div>

        <div className="idiom-detail-actions">
          <Button
            type={favorites.has(selectedIdiom.id) ? 'default' : 'primary'}
            icon={favorites.has(selectedIdiom.id) ? <StarFilled style={{ color: '#ffd700' }} /> : <StarOutlined />}
            onClick={() => toggleFavorite(selectedIdiom.id)}
          >
            {favorites.has(selectedIdiom.id) ? '取消收藏' : '添加收藏'}
          </Button>
        </div>
      </Card>
    )
  }

  // 渲染收藏的题目
  const renderFavoriteQuestions = () => {
    const favoriteList = idiomQuestions.filter(q => favorites.has(q.id))

    if (favoriteList.length === 0) {
      return <Empty description="暂无收藏的题目" />
    }

    return (
      <div className="favorites-list">
        {favoriteList.map(question => renderQuestionItem(question))}
      </div>
    )
  }

  // 渲染搜索历史
  const renderSearchHistory = () => {
    if (searchHistory.length === 0) {
      return <Empty description="暂无搜索历史" />
    }

    return (
      <div className="search-history-list">
        <div className="history-header">
          <h4>搜索历史</h4>
          <Button type="text" size="small" onClick={clearSearchHistory}>清除</Button>
        </div>
        <div className="history-items">
          {searchHistory.map((item, index) => (
            <Tag
              key={index}
              color="blue"
              className="history-item-tag"
              onClick={() => handleHistoryClick(item)}
              closable
              onClose={() => {
                const newHistory = searchHistory.filter((_, i) => i !== index)
                setSearchHistory(newHistory)
                saveSearchHistory(newHistory)
              }}
            >
              {item}
            </Tag>
          ))}
        </div>
      </div>
    )
  }

  // 渲染练习模式
  const renderExercise = () => {
    if (!exerciseMode) {
      return (
        <Card className="exercise-start-card">
          <div className="exercise-start-content">
            <h3>成语练习</h3>
            <p>通过练习巩固您对成语的理解和运用能力</p>
            <Button type="primary" size="large" onClick={startExercise}>
              开始练习
            </Button>
          </div>
        </Card>
      )
    }

    if (currentQuestionIndex >= idiomQuestions.length) {
      return (
        <Card className="exercise-summary-card">
          <div className="exercise-summary-content">
            <h3>练习完成</h3>
            <p>正确率：{Math.round((correctCount / answeredQuestions.length) * 100)}%</p>
            <p>答对题数：{correctCount} / {answeredQuestions.length}</p>
            <Button type="primary" onClick={() => { setExerciseMode(false); setCurrentTab('1') }}>
              返回
            </Button>
          </div>
        </Card>
      )
    }

    const currentQuestion = idiomQuestions[currentQuestionIndex]

    return (
      <Card className="exercise-card">
        <div className="exercise-header">
          <span className="question-progress">
            第 {currentQuestionIndex + 1} 题 / 共 {idiomQuestions.length} 题
          </span>
          <span className="progress-score">
            已答对: {correctCount} 题
          </span>
        </div>

        <div className="exercise-content">
          <h3 className="exercise-question">{currentQuestion.question}</h3>

          <div className="exercise-options">
            {currentQuestion.type === 'single_choice' ? (
              <Radio.Group
                value={selectedAnswer}
                onChange={(e) => setSelectedAnswer(e.target.value)}
                disabled={showExplanation}
              >
                <Space direction="vertical" style={{ width: '100%' }}>
                  {currentQuestion.options.map((option, index) => (
                    <Radio key={index} value={index}>
                      {String.fromCharCode(65 + index)}. {option}
                    </Radio>
                  ))}
                </Space>
              </Radio.Group>
            ) : (
              <Checkbox.Group
                value={selectedAnswer}
                onChange={setSelectedAnswer}
                disabled={showExplanation}
              >
                <Space direction="vertical" style={{ width: '100%' }}>
                  {currentQuestion.options.map((option, index) => (
                    <Checkbox key={index} value={index}>
                      {String.fromCharCode(65 + index)}. {option}
                    </Checkbox>
                  ))}
                </Space>
              </Checkbox.Group>
            )}
          </div>

          <div className="exercise-actions">
            {!showExplanation ? (
              <Button
                type="primary"
                onClick={submitAnswer}
                disabled={selectedAnswer === null}
              >
                提交答案
              </Button>
            ) : (
              <Button type="primary" onClick={nextQuestion}>
                {currentQuestionIndex < idiomQuestions.length - 1 ? '下一题' : '完成练习'}
              </Button>
            )}
          </div>

          {exerciseResult && showExplanation && (
            <div className={`exercise-result ${exerciseResult.isCorrect ? 'correct' : 'incorrect'}`}>
              <div className="result-icon">
                {exerciseResult.isCorrect ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
              </div>
              <div className="result-text">
                {exerciseResult.isCorrect ? '回答正确！' : '回答错误'}
              </div>
              {!exerciseResult.isCorrect && (
                <div className="correct-answer">
                  正确答案：{currentQuestion.options[exerciseResult.correctAnswer]}
                </div>
              )}
            </div>
          )}

          {showExplanation && exerciseResult.explanation && (
            <div className="explanation-section">
              <h4>解析：</h4>
              <p>{exerciseResult.explanation}</p>
            </div>
          )}
        </div>
      </Card>
    )
  }

  // 主内容渲染
  return (
    <div className="idiom-container">
      <h1 className="page-title">熟语习语学习</h1>

      {/* 搜索区域 */}
      <div className="search-filter-section">
        <Search
          placeholder="搜索题目内容或选项"
          allowClear
          enterButton={<SearchOutlined />}
          size="large"
          onSearch={handleSearch}
          onChange={(e) => setSearchText(e.target.value)}
          value={searchText}
          className="search-input"
        />
      </div>

      {/* 内容标签页 */}
      <Tabs activeKey={currentTab} onChange={setCurrentTab} className="idiom-tabs">
        <TabPane tab="题目列表" key="1">
          {loading ? (
            <div className="loading-wrapper">
              <Spin size="large" tip="加载中..." />
            </div>
          ) : filteredQuestions.length > 0 ? (
            <div className="idioms-grid">
              {filteredQuestions.map(renderQuestionItem)}
            </div>
          ) : (
            <Empty description="没有找到匹配的题目" />
          )}
        </TabPane>

        <TabPane tab="题目详情" key="2">
          {renderQuestionDetail()}
        </TabPane>

        <TabPane tab="我的收藏" key="3">
          {renderFavoriteQuestions()}
        </TabPane>

        <TabPane tab="搜索历史" key="4">
          {renderSearchHistory()}
        </TabPane>

        <TabPane tab="成语练习" key="5">
          {renderExercise()}
        </TabPane>
      </Tabs>
    </div>
  )
}

export default Idiom