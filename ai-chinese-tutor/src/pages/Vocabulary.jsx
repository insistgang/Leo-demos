import React, { useState, useEffect } from 'react'
import { Card, Input, Button, Tabs, List, Divider, message, Spin, Tooltip, Modal, Tag, Avatar, Badge, Empty } from 'antd'
import { SearchOutlined, SoundOutlined, StarOutlined, StarFilled, BookOutlined, FileTextOutlined, ReloadOutlined, HistoryOutlined } from '@ant-design/icons'
import { vocabularyService } from '../services/vocabularyService'
import { studyRecordService } from '../services/studyRecordService'
import { questionService } from '../services/questionService'

const { TabPane } = Tabs
const { TextArea } = Input

const Vocabulary = () => {
  const [searchWord, setSearchWord] = useState('')
  const [wordDetail, setWordDetail] = useState(null)
  const [relatedWords, setRelatedWords] = useState({ synonyms: [], antonyms: [], phrases: [], examples: [] })
  const [idioms, setIdioms] = useState([])
  const [loading, setLoading] = useState(false)
  const [currentTab, setCurrentTab] = useState('1')
  const [favoriteWords, setFavoriteWords] = useState([])
  const [searchHistory, setSearchHistory] = useState([])
  const [recentWords, setRecentWords] = useState([])
  const [practiceData, setPracticeData] = useState(null)
  const [isPracticeStarted, setIsPracticeStarted] = useState(false)
  const [currentQuestion, setCurrentQuestion] = useState(null)
  const [practiceResult, setPracticeResult] = useState({ correct: 0, total: 0 })
  const [loadingPractice, setLoadingPractice] = useState(false)
  const [showFeedback, setShowFeedback] = useState(false)
  const [selectedAnswer, setSelectedAnswer] = useState(null)
  const [practiceQuestions, setPracticeQuestions] = useState([])
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
  
  // 从本地存储加载收藏词和历史记录
  useEffect(() => {
    loadFavoriteWords()
    loadSearchHistory()
    loadRecentWords()
    loadIdioms()
  }, [])
  
  const loadFavoriteWords = () => {
    const favorites = localStorage.getItem('favoriteWords')
    if (favorites) {
      setFavoriteWords(JSON.parse(favorites))
    }
  }
  
  const loadSearchHistory = () => {
    const history = localStorage.getItem('searchHistory')
    if (history) {
      setSearchHistory(JSON.parse(history))
    }
  }
  
  const loadRecentWords = () => {
    const recent = localStorage.getItem('recentWords')
    if (recent) {
      setRecentWords(JSON.parse(recent))
    }
  }
  
  const loadIdioms = async () => {
    try {
      // 模拟获取成语列表
      const mockIdioms = [
        { id: 1, word: '一举两得', pinyin: 'yī jǔ liǎng dé', explanation: '做一件事得到两方面的好处' },
        { id: 2, word: '三心二意', pinyin: 'sān xīn èr yì', explanation: '又想这样又想那样，犹豫不定' },
        { id: 3, word: '四面八方', pinyin: 'sì miàn bā fāng', explanation: '各个方面或各个地方' },
        { id: 4, word: '五光十色', pinyin: 'wǔ guāng shí sè', explanation: '形容色泽鲜艳，花样繁多' },
        { id: 5, word: '六神无主', pinyin: 'liù shén wú zhǔ', explanation: '形容惊慌着急，没了主意，不知如何才好' }
      ]
      setIdioms(mockIdioms)
    } catch (error) {
      message.error('加载成语列表失败')
    }
  }
  
  const handleSearch = async () => {
    if (!searchWord.trim()) {
      message.warning('请输入要查询的词语')
      return
    }
    
    setLoading(true)
    try {
      // 模拟查询词语
      const mockWordDetail = {
          word: searchWord,
          pinyin: 'mó nǐ',
          pronunciation: 'https://example.com/pronunciation.mp3',
          definitions: [
            '模仿现成的样子做。',
            '指对外界事物的一种仿真反应。'
          ],
          partOfSpeech: '动词',
          radical: '木',
          strokeCount: 15,
          collocations: ['模仿能力', '模仿创新', '模仿学习'],
          usageNotes: '多用于书面语，可带宾语或补语。',
          examples: [
            '他善于模仿各种动物的叫声。',
            '这个软件可以模拟真实的物理环境。'
          ],
          origin: '出自《后汉书·吕强传》' 
        }
      
      const mockRelatedWords = {
          synonyms: ['仿照', '模拟', '仿效'],
          antonyms: ['创新', '创造', '原创'],
          hypernyms: ['效仿', '借鉴'],
          hyponyms: ['仿制', '仿造', '仿照'],
          themes: ['学习方法', '技能培养'],
          phrases: ['模拟考试', '模拟训练', '模拟环境'],
          examples: [
            '通过模拟实验，我们得出了准确的数据。',
            '模拟飞行训练可以提高飞行员的应急反应能力。'
          ]
        }
      
      setWordDetail(mockWordDetail)
      setRelatedWords(mockRelatedWords)
      
      // 添加到搜索历史
      addToSearchHistory(searchWord)
      // 添加到最近查看
      addToRecentWords(mockWordDetail)
    } catch (error) {
      message.error('查询失败：' + error.message)
    } finally {
      setLoading(false)
    }
  }
  
  const addToSearchHistory = (word) => {
    let history = [...searchHistory]
    if (history.includes(word)) {
      history = history.filter(w => w !== word)
    }
    history.unshift(word)
    if (history.length > 10) {
      history = history.slice(0, 10)
    }
    setSearchHistory(history)
    localStorage.setItem('searchHistory', JSON.stringify(history))
  }
  
  const addToRecentWords = (word) => {
    let recent = [...recentWords]
    const exists = recent.findIndex(w => w.word === word.word) !== -1
    if (exists) {
      recent = recent.filter(w => w.word !== word.word)
    }
    recent.unshift(word)
    if (recent.length > 20) {
      recent = recent.slice(0, 20)
    }
    setRecentWords(recent)
    localStorage.setItem('recentWords', JSON.stringify(recent))
  }
  
  const toggleFavorite = (word) => {
    let favorites = [...favoriteWords]
    const index = favorites.findIndex(f => f.word === word.word)
    if (index !== -1) {
      favorites.splice(index, 1)
      message.success('已取消收藏')
    } else {
      favorites.push(word)
      message.success('收藏成功')
    }
    setFavoriteWords(favorites)
    localStorage.setItem('favoriteWords', JSON.stringify(favorites))
  }
  
  const playPronunciation = () => {
    if (wordDetail?.pronunciation) {
      // 这里只是模拟播放，实际项目中需要使用HTML5 Audio API
      message.success('播放发音')
    }
  }
  
  const searchFromHistory = (word) => {
    setSearchWord(word)
    // 触发搜索
    setLoading(true)
    setTimeout(() => {
      handleSearch()
    }, 100)
  }
  
  const clearSearchHistory = () => {
    Modal.confirm({
      title: '确认清除',
      content: '确定要清除所有搜索历史吗？',
      onOk: () => {
        setSearchHistory([])
        localStorage.removeItem('searchHistory')
        message.success('搜索历史已清除')
      }
    })
  }
  
  const startPractice = async () => {
    setLoadingPractice(true)
    try {
      // 从数据库获取词汇练习题
      const questions = await questionService.getQuestionsByType('vocabulary', { count: 10 })

      if (questions.length === 0) {
        message.warning('暂无词汇练习题，请稍后再试')
        setLoadingPractice(false)
        return
      }

      setPracticeQuestions(questions)
      setCurrentQuestionIndex(0)
      setCurrentQuestion(questions[0])
      setIsPracticeStarted(true)
      setPracticeResult({ correct: 0, total: questions.length })
      setShowFeedback(false)
      setSelectedAnswer(null)
    } catch (error) {
      console.error('加载练习题失败:', error)
      message.error('加载练习题失败，请稍后重试')
    } finally {
      setLoadingPractice(false)
    }
  }
  
  const selectAnswer = async (index) => {
    if (!currentQuestion || showFeedback) return

    setSelectedAnswer(index)
    const isCorrect = index === currentQuestion.correctAnswer

    // 立即保存答题记录到数据库
    try {
      await saveAnswerToDatabase(currentQuestion.id, index, isCorrect)
    } catch (error) {
      console.error('保存答题记录失败:', error)
    }

    // 更新得分
    if (isCorrect) {
      setPracticeResult(prev => ({ ...prev, correct: prev.correct + 1 }))
      message.success('回答正确！+10分')
    } else {
      message.error('回答错误')
    }

    // 显示解释
    setShowFeedback(true)
  }

  // 保存单个答题记录到数据库
  const saveAnswerToDatabase = async (questionId, selectedAnswer, isCorrect) => {
    try {
      const recordData = {
        type: 'vocabulary',
        content: {
          questionId: questionId,
          selectedAnswer: selectedAnswer,
          isCorrect: isCorrect,
          difficulty: currentQuestion?.difficulty || '中等'
        },
        duration: 0,
        score: isCorrect ? 10 : 0,
        totalQuestions: 1,
        correctAnswers: isCorrect ? 1 : 0
      }

      const saveResult = await studyRecordService.addStudyRecord(recordData)
      if (!saveResult.success) {
        console.error('答题记录保存失败:', saveResult.message)
      }
    } catch (error) {
      console.error('保存答题记录出错:', error)
    }
  }
  
  const nextQuestion = () => {
    if (!practiceQuestions || practiceQuestions.length === 0) return

    if (currentQuestionIndex < practiceQuestions.length - 1) {
      // 下一题
      const nextIndex = currentQuestionIndex + 1
      setCurrentQuestionIndex(nextIndex)
      setCurrentQuestion(practiceQuestions[nextIndex])
      setShowFeedback(false)
      setSelectedAnswer(null)
    } else {
      // 练习完成
      finishPractice()
    }
  }

  // 完成练习
  const finishPractice = async () => {
    setIsPracticeStarted(false)

    // 保存练习总结
    try {
      await savePracticeSummary()
    } catch (error) {
      console.error('保存练习总结失败:', error)
    }

    // 显示完成消息
    const accuracy = Math.round((practiceResult.correct / practiceResult.total) * 100)
    Modal.success({
      title: '练习完成！',
      content: (
        <div>
          <p>答对题数：{practiceResult.correct} / {practiceResult.total}</p>
          <p>正确率：{accuracy}%</p>
          <p>得分：{practiceResult.correct * 10} 分</p>
        </div>
      ),
      okText: '确定'
    })
  }

  // 保存练习总结
  const savePracticeSummary = async () => {
    try {
      const recordData = {
        type: 'vocabulary',
        content: {
          summary: '词汇练习完成',
          totalQuestions: practiceResult.total,
          correctAnswers: practiceResult.correct,
          accuracy: Math.round((practiceResult.correct / practiceResult.total) * 100)
        },
        duration: 0, // TODO: 计算实际用时
        score: practiceResult.correct * 10,
        totalQuestions: practiceResult.total,
        correctAnswers: practiceResult.correct
      }

      const saveResult = await studyRecordService.addStudyRecord(recordData)
      if (saveResult.success) {
        console.log('词汇练习总结保存成功')
      } else {
        console.error('词汇练习总结保存失败:', saveResult.message)
      }
    } catch (error) {
      console.error('保存词汇练习总结出错:', error)
    }
  }

  const isWordFavorite = (word) => {
    return favoriteWords.some(f => f.word === word)
  }
  
  return (
    <div className="vocabulary-container">
      <Card className="search-card">
        <div className="search-input-wrapper">
          <Input
            placeholder="请输入要查询的词语"
            value={searchWord}
            onChange={(e) => setSearchWord(e.target.value)}
            onPressEnter={handleSearch}
            className="search-input"
            prefix={<SearchOutlined />}
          />
          <Button type="primary" onClick={handleSearch} className="search-btn">
            查询
          </Button>
        </div>
      </Card>
      
      {loading ? (
        <div className="loading-wrapper">
          <Spin size="large" tip="查询中..." />
        </div>
      ) : (
        <Tabs activeKey={currentTab} onChange={setCurrentTab} className="vocabulary-tabs">
          <TabPane tab="词语详情" key="1">
            {wordDetail ? (
              <div className="word-detail">
                <div className="word-header">
                  <h2>{wordDetail.word}</h2>
                  <span className="pinyin">{wordDetail.pinyin}</span>
                  <Tooltip title="播放发音">
        <Button 
          type="text" 
          icon={<SoundOutlined />} 
          onClick={playPronunciation}
          className="pronunciation-btn"
        />
      </Tooltip>
                  <Tooltip title={isWordFavorite(wordDetail.word) ? '取消收藏' : '收藏'}>
                    <Button 
                      type="text" 
                      icon={isWordFavorite(wordDetail.word) ? <StarFilled /> : <StarOutlined />}
                      onClick={() => toggleFavorite(wordDetail)}
                      className="favorite-btn"
                      style={{ color: isWordFavorite(wordDetail.word) ? '#ff4d4f' : '' }}
                    />
                  </Tooltip>
                </div>
                
                <Divider />
                
                <div className="word-info">
                  {wordDetail.partOfSpeech && (
                    <div className="info-item">
                      <Tag color="blue">{wordDetail.partOfSpeech}</Tag>
                    </div>
                  )}
                  
                  {wordDetail.radical && (
                    <div className="info-item">
                      <span className="info-label">部首：</span>
                      <span className="info-value">{wordDetail.radical}</span>
                    </div>
                  )}
                  
                  {wordDetail.strokeCount && (
                    <div className="info-item">
                      <span className="info-label">笔画：</span>
                      <span className="info-value">{wordDetail.strokeCount}</span>
                    </div>
                  )}
                </div>
                
                <Divider />
                
                <div className="definitions">
                  <h3>释义</h3>
                  <List
                    dataSource={wordDetail.definitions}
                    renderItem={(item, index) => (
                      <List.Item className="definition-item">
                        <span className="definition-number">{index + 1}.</span>
                        <span className="definition-text">{item}</span>
                      </List.Item>
                    )}
                  />
                </div>
                
                {wordDetail.examples && wordDetail.examples.length > 0 && (
                  <div className="examples">
                    <h3>例句</h3>
                    <List
                      dataSource={wordDetail.examples}
                      renderItem={(item, index) => (
                        <List.Item className="example-item">
                          <span className="example-text">{item}</span>
                        </List.Item>
                      )}
                    />
                  </div>
                )}
                
                {wordDetail.collocations && wordDetail.collocations.length > 0 && (
                  <div className="collocations">
                    <h3>常用搭配</h3>
                    <div className="collocation-tags">
                      {wordDetail.collocations.map((collocation, index) => (
                        <Tag key={index} color="purple">{collocation}</Tag>
                      ))}
                    </div>
                  </div>
                )}

                {wordDetail.usageNotes && (
                  <div className="usage-notes">
                    <h3>用法说明</h3>
                    <p>{wordDetail.usageNotes}</p>
                  </div>
                )}

                {wordDetail.origin && (
                  <div className="origin">
                    <h3>出处</h3>
                    <p>{wordDetail.origin}</p>
                  </div>
                )}
              </div>
            ) : (
              <Empty description="请输入词语进行查询" />
            )}
          </TabPane>
          
          <TabPane tab="相关词语" key="2">
            {relatedWords.synonyms.length > 0 && (
              <div className="related-words-section">
                <h3>近义词</h3>
                <div className="related-words-tags">
                  {relatedWords.synonyms.map((word, index) => (
                    <Tag key={index} color="green" className="related-word-tag">{word}</Tag>
                  ))}
                </div>
              </div>
            )}
            
            {relatedWords.antonyms.length > 0 && (
              <div className="related-words-section">
                <h3>反义词</h3>
                <div className="related-words-tags">
                  {relatedWords.antonyms.map((word, index) => (
                    <Tag key={index} color="red" className="related-word-tag">{word}</Tag>
                  ))}
                </div>
              </div>
            )}

            {relatedWords.hypernyms && relatedWords.hypernyms.length > 0 && (
              <div className="related-words-section">
                <h3>上位词</h3>
                <div className="related-words-tags">
                  {relatedWords.hypernyms.map((word, index) => (
                    <Tag key={index} color="orange" className="related-word-tag">{word}</Tag>
                  ))}
                </div>
              </div>
            )}

            {relatedWords.hyponyms && relatedWords.hyponyms.length > 0 && (
              <div className="related-words-section">
                <h3>下位词</h3>
                <div className="related-words-tags">
                  {relatedWords.hyponyms.map((word, index) => (
                    <Tag key={index} color="cyan" className="related-word-tag">{word}</Tag>
                  ))}
                </div>
              </div>
            )}

            {relatedWords.themes && relatedWords.themes.length > 0 && (
              <div className="related-words-section">
                <h3>主题词汇</h3>
                <div className="related-words-tags">
                  {relatedWords.themes.map((theme, index) => (
                    <Tag key={index} color="geekblue" className="related-word-tag">{theme}</Tag>
                  ))}
                </div>
              </div>
            )}

            {relatedWords.phrases.length > 0 && (
              <div className="related-words-section">
                <h3>相关词组</h3>
                <div className="related-words-tags">
                  {relatedWords.phrases.map((phrase, index) => (
                    <Tag key={index} color="blue" className="related-word-tag">{phrase}</Tag>
                  ))}
                </div>
              </div>
            )}
            
            {relatedWords.examples.length > 0 && (
              <div className="related-words-section">
                <h3>用法示例</h3>
                <List
                  dataSource={relatedWords.examples}
                  renderItem={(item, index) => (
                    <List.Item className="example-item">
                      <span className="example-text">{item}</span>
                    </List.Item>
                  )}
                />
              </div>
            )}
            
            {relatedWords.synonyms.length === 0 && 
             relatedWords.antonyms.length === 0 && 
             relatedWords.phrases.length === 0 && 
             relatedWords.examples.length === 0 && (
              <Empty description="暂无相关词语" />
            )}
          </TabPane>
          
          <TabPane tab="成语学习" key="3">
            <div className="idioms-section">
              <List
                dataSource={idioms}
                renderItem={(idiom) => (
                  <List.Item className="idiom-item">
                    <div className="idiom-content">
                      <h4>{idiom.word}</h4>
                      <p className="idiom-pinyin">{idiom.pinyin}</p>
                      <p className="idiom-explanation">{idiom.explanation}</p>
                    </div>
                  </List.Item>
                )}
              />
            </div>
          </TabPane>
          
          <TabPane tab="练习测试" key="4">
            <div className="practice-section">
              {!isPracticeStarted ? (
                <div className="practice-start">
                  <h3>词汇练习</h3>
                  <p>通过练习巩固你的词汇知识</p>
                  <Button type="primary" size="large" onClick={startPractice} loading={loadingPractice}>
                    开始练习
                  </Button>
                  {practiceResult.total > 0 && (
                    <div style={{ marginTop: 20 }}>
                      <p>上次练习成绩：{practiceResult.correct}/{practiceResult.total} 题</p>
                      <p>得分：{practiceResult.correct * 10} 分</p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="practice-content">
                  {loadingPractice ? (
                    <div style={{ textAlign: 'center', padding: '50px' }}>
                      <Spin size="large" tip="加载题目中..." />
                    </div>
                  ) : currentQuestion ? (
                    <div className="question-container">
                      <h3>问题 {currentQuestionIndex + 1}/{practiceQuestions.length}</h3>
                      <p className="question-text">{currentQuestion.question}</p>
                      {currentQuestion.source && (
                        <p style={{ color: '#666', fontSize: '12px' }}>来源：{currentQuestion.source}</p>
                      )}

                      <div className="options">
                        {currentQuestion.options.map((option, index) => {
                          const isSelected = selectedAnswer === index
                          const isCorrect = index === currentQuestion.correctAnswer
                          const showCorrect = showFeedback && isCorrect
                          const showIncorrect = showFeedback && isSelected && !isCorrect

                          return (
                            <Button
                              key={index}
                              type={
                                showCorrect ? 'primary' :
                                showIncorrect ? 'danger' :
                                isSelected ? 'default' : 'default'
                              }
                              className={`option-btn ${
                                showCorrect ? 'correct' :
                                showIncorrect ? 'incorrect' :
                                ''
                              }`}
                              onClick={() => selectAnswer(index)}
                              disabled={showFeedback}
                              block
                              style={{
                                marginBottom: '10px',
                                textAlign: 'left',
                                height: 'auto',
                                padding: '10px 15px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'flex-start',
                                backgroundColor: showCorrect ? '#f6ffed' : showIncorrect ? '#fff2f0' : ''
                              }}
                            >
                              <span style={{ marginRight: '10px', fontWeight: 'bold' }}>
                                {String.fromCharCode(65 + index)}.
                              </span>
                              <span>{option}</span>
                              {showCorrect && <span style={{ marginLeft: 'auto', color: '#52c41a' }}>✓</span>}
                              {showIncorrect && <span style={{ marginLeft: 'auto', color: '#ff4d4f' }}>✗</span>}
                            </Button>
                          )
                        })}
                      </div>

                      {showFeedback && (
                        <div style={{ marginTop: '20px', padding: '15px', backgroundColor: '#f0f2f5', borderRadius: '6px' }}>
                          <h4 style={{ marginBottom: '10px', color: selectedAnswer === currentQuestion.correctAnswer ? '#52c41a' : '#ff4d4f' }}>
                            {selectedAnswer === currentQuestion.correctAnswer ? '回答正确！' : '回答错误'}
                          </h4>
                          {currentQuestion.explanation && (
                            <div>
                              <p style={{ fontWeight: 'bold', marginBottom: '5px' }}>解析：</p>
                              <p>{currentQuestion.explanation}</p>
                            </div>
                          )}
                        </div>
                      )}

                      {showFeedback && (
                        <Button
                          type="primary"
                          onClick={nextQuestion}
                          className="next-btn"
                          style={{ marginTop: '20px' }}
                        >
                          {currentQuestionIndex < practiceQuestions.length - 1 ? '下一题' : '完成练习'}
                        </Button>
                      )}
                    </div>
                  ) : (
                    <Empty description="没有找到题目" />
                  )}
                </div>
              )}
            </div>
          </TabPane>
          
          <TabPane tab="我的收藏" key="5">
            <div className="favorite-section">
              {favoriteWords.length > 0 ? (
                <List
                  dataSource={favoriteWords}
                  renderItem={(word) => (
                    <List.Item 
                      className="favorite-item" 
                      actions={[
                        <Button type="text" icon={<StarFilled />} onClick={() => toggleFavorite(word)} style={{ color: '#ff4d4f' }}>
                          取消收藏
                        </Button>
                      ]}
                    >
                      <List.Item.Meta
                        avatar={<Avatar>{word.word.substring(0, 1)}</Avatar>}
                        title={word.word}
                        description={word.pinyin}
                      />
                    </List.Item>
                  )}
                />
              ) : (
                <Empty description="暂无收藏的词语" />
              )}
            </div>
          </TabPane>
          
          <TabPane tab="搜索历史" key="6">
            <div className="history-section">
              {searchHistory.length > 0 ? (
                <>
                  <div className="history-header">
                    <h3>搜索历史</h3>
                    <Button type="text" icon={<ReloadOutlined />} onClick={clearSearchHistory}>
                      清除历史
                    </Button>
                  </div>
                  <List
                    dataSource={searchHistory}
                    renderItem={(word) => (
                      <List.Item 
                        className="history-item" 
                        onClick={() => searchFromHistory(word)}
                      >
                        <HistoryOutlined className="history-icon" />
                        <span className="history-word">{word}</span>
                      </List.Item>
                    )}
                  />
                </>
              ) : (
                <Empty description="暂无搜索历史" />
              )}
            </div>
          </TabPane>
        </Tabs>
      )}
    </div>
  )
}

export default Vocabulary