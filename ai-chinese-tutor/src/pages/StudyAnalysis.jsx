import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Tabs, Statistic, Divider, Empty, Button, Table, Tag, Tooltip } from 'antd'
import { BarChartOutlined, LineChartOutlined, PieChartOutlined, CheckCircleOutlined, ExclamationCircleOutlined, BookOutlined, TrophyOutlined, CalendarOutlined, ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons'
import { Bar, Line, Pie } from '@ant-design/plots'
import { studyRecordService } from '../services/studyRecordService'

const StudyAnalysis = () => {
  // 状态管理
  const [loading, setLoading] = useState(false)
  const [analysisData, setAnalysisData] = useState({
    strengths: [],
    weaknesses: [],
    improvementAreas: []
  })
  const [learningTrends, setLearningTrends] = useState([])
  const [masteryData, setMasteryData] = useState([])
  const [exerciseAnalysis, setExerciseAnalysis] = useState([])
  const [knowledgeGaps, setKnowledgeGaps] = useState([])
  const [recommendations, setRecommendations] = useState([])
  const [activeTabKey, setActiveTabKey] = useState('1')
  
  // 初始化数据
  useEffect(() => {
    loadAnalysisData()
  }, [])
  
  // 加载分析数据
  const loadAnalysisData = async () => {
    setLoading(true)
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // 加载学习分析数据
      const analysis = await loadLearningAnalysis()
      setAnalysisData(analysis)
      
      // 加载学习趋势数据
      const trends = await loadLearningTrends()
      setLearningTrends(trends)
      
      // 加载掌握度数据
      const mastery = await loadMasteryData()
      console.log('设置掌握度数据:', mastery)
      setMasteryData(mastery)
      
      // 加载练习分析数据
      const exercise = await loadExerciseAnalysis()
      setExerciseAnalysis(exercise)
      
      // 加载知识缺口数据
      const gaps = await loadKnowledgeGaps()
      setKnowledgeGaps(gaps)
      
      // 加载学习建议
      const recs = await loadRecommendations()
      setRecommendations(recs)
      
    } catch (error) {
      console.error('加载学习分析数据失败:', error)
    } finally {
      setLoading(false)
    }
  }
  
  // 加载学习优劣势分析
  const loadLearningAnalysis = async () => {
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await studyRecordService.getLearningAnalysis()
      
      // 模拟分析数据
      return {
        strengths: [
          '词语记忆能力强，记忆速度快',
          '成语理解能力较好，能正确使用大部分成语',
          '学习态度积极，每周学习频率稳定',
          '阅读理解能力有明显提升'
        ],
        weaknesses: [
          '语法知识掌握不够扎实',
          '词语辨析能力有待提高',
          '写作能力需要加强',
          '知识点容易遗忘，需要加强复习'
        ],
        improvementAreas: [
          '增加语法练习的频率',
          '使用记忆技巧加强词语辨析能力',
          '每周安排一定时间进行写作练习',
          '建立系统的复习计划'
        ]
      }
    } catch (error) {
      console.error('加载学习分析失败:', error)
      return {
        strengths: [],
        weaknesses: [],
        improvementAreas: []
      }
    }
  }
  
  // 加载学习趋势数据
  const loadLearningTrends = async () => {
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await studyRecordService.getLearningTrends()
      
      // 模拟学习趋势数据（近8周）
      return [
        { week: '第1周', score: 65, duration: 120, wordsLearned: 50 },
        { week: '第2周', score: 68, duration: 135, wordsLearned: 65 },
        { week: '第3周', score: 72, duration: 150, wordsLearned: 75 },
        { week: '第4周', score: 70, duration: 140, wordsLearned: 70 },
        { week: '第5周', score: 76, duration: 165, wordsLearned: 85 },
        { week: '第6周', score: 78, duration: 170, wordsLearned: 90 },
        { week: '第7周', score: 82, duration: 180, wordsLearned: 100 },
        { week: '第8周', score: 85, duration: 195, wordsLearned: 110 }
      ]
    } catch (error) {
      console.error('加载学习趋势失败:', error)
      return []
    }
  }
  
  // 加载掌握度数据
  const loadMasteryData = async () => {
    try {
      // 获取学习统计数据
      const statistics = await studyRecordService.getStudyStatistics(365)

      // 根据实际数据计算掌握度
      const masteryData = []

      // 如果有分类数据，使用实际数据
      if (statistics.categoryData && statistics.categoryData.length > 0) {
        statistics.categoryData.forEach(category => {
          let masteryLevel = Math.min(100, category.studyTime / 6) // 假设6小时=100%掌握
          if (category.count > 0) {
            // 根据练习次数调整掌握度
            masteryLevel = Math.min(100, masteryLevel + (category.count * 2))
          }

          masteryData.push({
            category: category.category,
            masteryLevel: Math.round(masteryLevel),
            color: getColorForCategory(category.category)
          })
        })
      }

      // 确保包含所有主要类别
      const defaultCategories = [
        { category: '词语学习', color: '#1890ff' },
        { category: '成语学习', color: '#52c41a' },
        { category: '古诗词', color: '#faad14' },
        { category: '拼音', color: '#f5222d' },
        { category: '文本纠错', color: '#722ed1' }
      ]

      defaultCategories.forEach(defaultCat => {
        if (!masteryData.find(item => item.category === defaultCat.category)) {
          masteryData.push({
            category: defaultCat.category,
            masteryLevel: 0,
            color: defaultCat.color
          })
        }
      })

      console.log('掌握度数据:', masteryData)
      return masteryData
    } catch (error) {
      console.error('加载掌握度数据失败:', error)
      return []
    }
  }

  // 获取分类对应的颜色
  const getColorForCategory = (category) => {
    const colorMap = {
      '词语学习': '#1890ff',
      '成语学习': '#52c41a',
      '古诗词': '#faad14',
      '拼音': '#f5222d',
      '文本纠错': '#722ed1',
      '练习': '#13c2c2'
    }
    return colorMap[category] || '#1890ff'
  }
  
  // 加载练习分析数据
  const loadExerciseAnalysis = async () => {
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await studyRecordService.getExerciseAnalysis()
      
      // 模拟练习分析数据
      return [
        { type: '选择题', correctRate: 85, totalAttempts: 120, averageTime: 25 },
        { type: '填空题', correctRate: 72, totalAttempts: 95, averageTime: 35 },
        { type: '判断题', correctRate: 90, totalAttempts: 85, averageTime: 15 },
        { type: '简答题', correctRate: 65, totalAttempts: 45, averageTime: 60 },
        { type: '写作题', correctRate: 60, totalAttempts: 20, averageTime: 120 }
      ]
    } catch (error) {
      console.error('加载练习分析数据失败:', error)
      return []
    }
  }
  
  // 加载知识缺口数据
  const loadKnowledgeGaps = async () => {
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await studyRecordService.getKnowledgeGaps()
      
      // 模拟知识缺口数据
      return [
        {
          id: 1,
          topic: '语法 - 复杂句子结构',
          errorRate: 75,
          attempts: 40,
          lastPracticed: '2025-08-08',
          suggestedPractice: '增加复杂句子分析练习'
        },
        {
          id: 2,
          topic: '词语 - 近义词辨析',
          errorRate: 68,
          attempts: 55,
          lastPracticed: '2025-08-09',
          suggestedPractice: '使用词语辨析卡片进行练习'
        },
        {
          id: 3,
          topic: '成语 - 生僻成语应用',
          errorRate: 62,
          attempts: 35,
          lastPracticed: '2025-08-05',
          suggestedPractice: '增加生僻成语的阅读和使用练习'
        },
        {
          id: 4,
          topic: '写作 - 议论文结构',
          errorRate: 70,
          attempts: 15,
          lastPracticed: '2025-08-02',
          suggestedPractice: '学习议论文范文并进行模仿写作'
        }
      ]
    } catch (error) {
      console.error('加载知识缺口数据失败:', error)
      return []
    }
  }
  
  // 加载学习建议
  const loadRecommendations = async () => {
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await studyRecordService.getLearningRecommendations()
      
      // 模拟学习建议数据
      return [
        {
          id: 1,
          title: '加强语法练习',
          description: '每周安排3次语法专项练习，重点关注复杂句子结构和病句修改。',
          priority: 'high',
          estimatedTime: '每次30分钟'
        },
        {
          id: 2,
          title: '建立词语复习计划',
          description: '使用间隔重复法复习已学词语，特别是近义词和易混淆词语。',
          priority: 'medium',
          estimatedTime: '每天15分钟'
        },
        {
          id: 3,
          title: '增加阅读量',
          description: '每天阅读至少30分钟的优秀作文或文学作品，积累好词好句。',
          priority: 'high',
          estimatedTime: '每天30分钟'
        },
        {
          id: 4,
          title: '练习写作',
          description: '每周完成1-2篇短文写作练习，重点提升议论文和记叙文写作能力。',
          priority: 'medium',
          estimatedTime: '每周60-90分钟'
        },
        {
          id: 5,
          title: '成语故事学习',
          description: '通过成语故事理解成语的来源和用法，提高成语应用能力。',
          priority: 'low',
          estimatedTime: '每周20分钟'
        }
      ]
    } catch (error) {
      console.error('加载学习建议失败:', error)
      return []
    }
  }
  
  // 生成学习趋势图配置
  const learningTrendConfig = {
    data: learningTrends,
    xField: 'week',
    yField: 'score',
    seriesField: 'week',
    yAxis: {
      label: {
        formatter: (v) => `${v}分`,
      },
      min: 0,
      max: 100
    },
    point: {
      size: 5,
      shape: 'circle',
    },
    tooltip: {
      formatter: (datum) => {
        return {
          name: datum.week,
          value: `${datum.score}分`,
        }
      },
    },
  }
  
  // 生成学习时长趋势图配置
  const durationTrendConfig = {
    data: learningTrends,
    xField: 'week',
    yField: 'duration',
    seriesField: 'week',
    yAxis: {
      label: {
        formatter: (v) => `${v}分钟`,
      },
    },
    point: {
      size: 5,
      shape: 'diamond',
    },
    tooltip: {
      formatter: (datum) => {
        return {
          name: datum.week,
          value: `${datum.duration}分钟`,
        }
      },
    },
  }
  
  // 生成掌握度饼图配置
  const masteryPieConfig = {
    data: masteryData,
    angleField: 'masteryLevel',
    colorField: 'category',
    radius: 0.8,
    label: {
      type: 'outer',
      content: '{name}: {percentage}',
    },
    interactions: [
      {
        type: 'element-active',
      },
    ],
    tooltip: {
      formatter: (datum) => {
        return {
          name: datum.category,
          value: `${datum.masteryLevel}%`,
        }
      },
    },
  }
  
  // 生成题型正确率柱状图配置
  const exerciseBarConfig = {
    data: exerciseAnalysis,
    xField: 'type',
    yField: 'correctRate',
    seriesField: 'type',
    yAxis: {
      label: {
        formatter: (v) => `${v}%`,
      },
      min: 0,
      max: 100
    },
    tooltip: {
      formatter: (datum) => {
        return {
          name: datum.type,
          value: `${datum.correctRate}%`,
        }
      },
    },
  }
  
  // 知识缺口表格列配置
  const knowledgeGapsColumns = [
    {
      title: '知识点',
      dataIndex: 'topic',
      key: 'topic',
    },
    {
      title: '错误率',
      dataIndex: 'errorRate',
      key: 'errorRate',
      render: (text) => (
        <Tag color={text > 70 ? 'red' : text > 50 ? 'orange' : 'yellow'}>
          {text}%
        </Tag>
      ),
      sorter: (a, b) => a.errorRate - b.errorRate,
    },
    {
      title: '练习次数',
      dataIndex: 'attempts',
      key: 'attempts',
      sorter: (a, b) => a.attempts - b.attempts,
    },
    {
      title: '最后练习',
      dataIndex: 'lastPracticed',
      key: 'lastPracticed',
      sorter: (a, b) => new Date(a.lastPracticed) - new Date(b.lastPracticed),
    },
    {
      title: '建议练习',
      dataIndex: 'suggestedPractice',
      key: 'suggestedPractice',
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Tooltip title="开始练习">
          <Button type="link" size="small">
            练习
          </Button>
        </Tooltip>
      ),
    },
  ]
  
  // 学习建议表格列配置
  const recommendationsColumns = [
    {
      title: '建议内容',
      dataIndex: 'title',
      key: 'title',
    },
    {
      title: '详细说明',
      dataIndex: 'description',
      key: 'description',
    },
    {
      title: '优先级',
      dataIndex: 'priority',
      key: 'priority',
      render: (text) => (
        <Tag color={text === 'high' ? 'red' : text === 'medium' ? 'orange' : 'green'}>
          {text === 'high' ? '高' : text === 'medium' ? '中' : '低'}
        </Tag>
      ),
      filters: [
        { text: '高', value: 'high' },
        { text: '中', value: 'medium' },
        { text: '低', value: 'low' },
      ],
      onFilter: (value, record) => record.priority === value,
    },
    {
      title: '预计用时',
      dataIndex: 'estimatedTime',
      key: 'estimatedTime',
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Tooltip title="采纳建议">
          <Button type="link" size="small">
            采纳
          </Button>
        </Tooltip>
      ),
    },
  ]
  
  // 获取优先级对应的图标
  const getPriorityIcon = (priority) => {
    if (priority === 'high') {
      return <ExclamationCircleOutlined style={{ color: '#f5222d' }} />
    } else if (priority === 'medium') {
      return <ExclamationCircleOutlined style={{ color: '#faad14' }} />
    } else {
      return <CheckCircleOutlined style={{ color: '#52c41a' }} />
    }
  }
  
  // 获取趋势图标
  const getTrendIcon = (value, previousValue) => {
    if (!previousValue) return null
    if (value > previousValue) {
      return <ArrowUpOutlined style={{ color: '#52c41a' }} />
    } else if (value < previousValue) {
      return <ArrowDownOutlined style={{ color: '#f5222d' }} />
    }
    return null
  }
  
  return (
    <div className="study-analysis-container">
      {/* 页面标题 */}
      <div className="analysis-header">
        <h1>学习分析</h1>
        <p className="subtitle">基于您的学习数据，为您提供个性化的学习分析和建议</p>
      </div>
      
      {/* 关键指标卡片 */}
      {learningTrends.length > 0 && (
        <Row gutter={16} className="key-indicators">
          <Col span={6}>
            <Card>
              <Statistic 
                title="当前平均得分" 
                value={learningTrends[learningTrends.length - 1]?.score || 0} 
                suffix="分" 
                valueStyle={{ color: '#1890ff' }}
                prefix={<TrophyOutlined />}
                extra={getTrendIcon(
                  learningTrends[learningTrends.length - 1]?.score,
                  learningTrends[learningTrends.length - 2]?.score
                )}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic 
                title="本周学习时长" 
                value={learningTrends[learningTrends.length - 1]?.duration || 0} 
                suffix="分钟" 
                valueStyle={{ color: '#52c41a' }}
                prefix={<CalendarOutlined />}
                extra={getTrendIcon(
                  learningTrends[learningTrends.length - 1]?.duration,
                  learningTrends[learningTrends.length - 2]?.duration
                )}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic 
                title="本周学习词语" 
                value={learningTrends[learningTrends.length - 1]?.wordsLearned || 0} 
                suffix="个" 
                valueStyle={{ color: '#faad14' }}
                prefix={<BookOutlined />}
                extra={getTrendIcon(
                  learningTrends[learningTrends.length - 1]?.wordsLearned,
                  learningTrends[learningTrends.length - 2]?.wordsLearned
                )}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic 
                title="总体掌握度" 
                value={masteryData.reduce((sum, item) => sum + item.masteryLevel, 0) / (masteryData.length || 1)} 
                suffix="%" 
                precision={1}
                valueStyle={{ color: '#f5222d' }}
                prefix={<PieChartOutlined />}
              />
            </Card>
          </Col>
        </Row>
      )}
      
      <Divider />
      
      {/* 分析内容选项卡 */}
      <Tabs activeKey={activeTabKey} onChange={setActiveTabKey}>
        {/* 学习优劣势分析 */}
        <Tabs.TabPane tab="学习优劣势" key="1">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="学习优势" loading={loading}>
                {analysisData.strengths.length > 0 ? (
                  <ul className="strengths-list">
                    {analysisData.strengths.map((strength, index) => (
                      <li key={index} className="strength-item">
                        <CheckCircleOutlined style={{ color: '#52c41a', marginRight: 8 }} />
                        {strength}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <Empty description="暂无数据" />
                )}
              </Card>
            </Col>
            <Col span={12}>
              <Card title="学习劣势" loading={loading}>
                {analysisData.weaknesses.length > 0 ? (
                  <ul className="weaknesses-list">
                    {analysisData.weaknesses.map((weakness, index) => (
                      <li key={index} className="weakness-item">
                        <ExclamationCircleOutlined style={{ color: '#f5222d', marginRight: 8 }} />
                        {weakness}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <Empty description="暂无数据" />
                )}
              </Card>
            </Col>
          </Row>
          
          <Row gutter={16} style={{ marginTop: 16 }}>
            <Col span={24}>
              <Card title="改进建议" loading={loading}>
                {analysisData.improvementAreas.length > 0 ? (
                  <ul className="improvement-list">
                    {analysisData.improvementAreas.map((area, index) => (
                      <li key={index} className="improvement-item">
                        <span className="improvement-number">{index + 1}.</span>
                        {area}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <Empty description="暂无数据" />
                )}
              </Card>
            </Col>
          </Row>
        </Tabs.TabPane>
        
        {/* 学习趋势分析 */}
        <Tabs.TabPane tab="学习趋势" key="2">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="得分趋势" loading={loading}>
                {learningTrends.length > 0 ? (
                  <Line {...learningTrendConfig} />
                ) : (
                  <Empty description="暂无数据" />
                )}
              </Card>
            </Col>
            <Col span={12}>
              <Card title="学习时长趋势" loading={loading}>
                {learningTrends.length > 0 ? (
                  <Line {...durationTrendConfig} />
                ) : (
                  <Empty description="暂无数据" />
                )}
              </Card>
            </Col>
          </Row>
        </Tabs.TabPane>
        
        {/* 掌握度分析 */}
        <Tabs.TabPane tab="掌握度分析" key="3">
          <Card title="知识点掌握度分布" loading={loading}>
            <div className="mastery-list">
              {masteryData.map((item) => (
                <div key={item.category} style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
                  <div style={{ width: 12, height: 12, backgroundColor: item.color, marginRight: 8 }}></div>
                  <span style={{ flex: 1 }}>{item.category}</span>
                  <span>{item.masteryLevel}%</span>
                </div>
              ))}
            </div>
          </Card>
        </Tabs.TabPane>
        
        {/* 练习分析 */}
        <Tabs.TabPane tab="练习分析" key="4">
          <Card title="题型正确率分析" loading={loading}>
            {exerciseAnalysis.length > 0 ? (
              <Bar {...exerciseBarConfig} />
            ) : (
              <Empty description="暂无数据" />
            )}
          </Card>
          
          <Card title="题型详细分析" loading={loading} style={{ marginTop: 16 }}>
            {exerciseAnalysis.length > 0 ? (
              <Table 
                columns={[
                  { title: '题型', dataIndex: 'type', key: 'type' },
                  { title: '正确率', dataIndex: 'correctRate', key: 'correctRate', render: (text) => `${text}%` },
                  { title: '练习次数', dataIndex: 'totalAttempts', key: 'totalAttempts' },
                  { title: '平均用时(秒)', dataIndex: 'averageTime', key: 'averageTime' }
                ]} 
                dataSource={exerciseAnalysis} 
                rowKey="type" 
                pagination={false}
              />
            ) : (
              <Empty description="暂无数据" />
            )}
          </Card>
        </Tabs.TabPane>
        
        {/* 知识缺口 */}
        <Tabs.TabPane tab="知识缺口" key="5">
          <Card title="需要加强的知识点" loading={loading}>
            {knowledgeGaps.length > 0 ? (
              <Table 
                columns={knowledgeGapsColumns} 
                dataSource={knowledgeGaps} 
                rowKey="id" 
                pagination={{ pageSize: 10 }}
              />
            ) : (
              <Empty description="暂无数据" />
            )}
          </Card>
        </Tabs.TabPane>
        
        {/* 学习建议 */}
        <Tabs.TabPane tab="个性化建议" key="6">
          <Card title="为您推荐的学习计划" loading={loading}>
            {recommendations.length > 0 ? (
              <Table 
                columns={recommendationsColumns} 
                dataSource={recommendations} 
                rowKey="id" 
                pagination={{ pageSize: 10 }}
              />
            ) : (
              <Empty description="暂无数据" />
            )}
          </Card>
        </Tabs.TabPane>
      </Tabs>
      
      {/* 底部操作按钮 */}
      <div className="analysis-footer" style={{ marginTop: 24, textAlign: 'center' }}>
        <Button type="primary" onClick={loadAnalysisData} loading={loading}>刷新分析数据</Button>
      </div>
    </div>
  )
}

export default StudyAnalysis