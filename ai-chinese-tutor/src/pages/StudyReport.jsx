import React, { useState, useEffect } from 'react'
import { Card, Row, Col, DatePicker, Select, Statistic, Divider, Tabs, Table, Empty, Button, Tooltip } from 'antd'
import { BookOutlined, TrophyOutlined, BarChartOutlined, LineChartOutlined, CalendarOutlined, DownloadOutlined, FilterOutlined } from '@ant-design/icons'
import { Line, Bar, Pie } from '@ant-design/plots'
import { studyRecordService } from '../services/studyRecordService'

const { RangePicker } = DatePicker
const { Option } = Select

const StudyReport = () => {
  // 状态管理
  const [timeRange, setTimeRange] = useState(null)
  const [reportType, setReportType] = useState('weekly') // weekly, monthly, custom
  const [loading, setLoading] = useState(false)
  const [studyStats, setStudyStats] = useState({
    totalDuration: 0,
    totalWords: 0,
    totalExercises: 0,
    averageScore: 0
  })
  const [weeklyData, setWeeklyData] = useState([])
  const [monthlyData, setMonthlyData] = useState([])
  const [categoryData, setCategoryData] = useState([])
  const [achievementData, setAchievementData] = useState([])
  const [detailedRecords, setDetailedRecords] = useState([])
  const [activeTabKey, setActiveTabKey] = useState('1')
  
  // 初始化数据
  useEffect(() => {
    loadReportData()
  }, [timeRange, reportType])
  
  // 加载报告数据
  const loadReportData = async () => {
    setLoading(true)
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 800))
      
      // 加载统计数据
      const stats = await loadStudyStats()
      setStudyStats(stats)
      
      // 根据报告类型加载不同的数据
      if (reportType === 'weekly') {
        const weekly = await loadWeeklyData()
        setWeeklyData(weekly)
      } else if (reportType === 'monthly') {
        const monthly = await loadMonthlyData()
        setMonthlyData(monthly)
      }
      
      // 加载分类数据
      const category = await loadCategoryData()
      setCategoryData(category)
      
      // 加载成就数据
      const achievements = await loadAchievementData()
      setAchievementData(achievements)
      
      // 加载详细记录
      const records = await loadDetailedRecords()
      setDetailedRecords(records)
      
    } catch (error) {
      console.error('加载报告数据失败:', error)
    } finally {
      setLoading(false)
    }
  }
  
  // 加载学习统计数据
  const loadStudyStats = async () => {
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await studyRecordService.getStudyStats(reportType, timeRange)
      
      // 模拟统计数据
      return {
        totalDuration: 1200,
        totalWords: 320,
        totalExercises: 15,
        averageScore: 85.5
      }
    } catch (error) {
      console.error('加载统计数据失败:', error)
      return {
        totalDuration: 0,
        totalWords: 0,
        totalExercises: 0,
        averageScore: 0
      }
    }
  }
  
  // 加载周学习数据
  const loadWeeklyData = async () => {
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await studyRecordService.getWeeklyStudyData()
      
      // 模拟周学习数据
      return [
        { day: '周一', duration: 120, words: 25, score: 85 },
        { day: '周二', duration: 180, words: 40, score: 88 },
        { day: '周三', duration: 90, words: 15, score: 78 },
        { day: '周四', duration: 210, words: 50, score: 92 },
        { day: '周五', duration: 150, words: 35, score: 86 },
        { day: '周六', duration: 240, words: 60, score: 94 },
        { day: '周日', duration: 180, words: 45, score: 90 }
      ]
    } catch (error) {
      console.error('加载周学习数据失败:', error)
      return []
    }
  }
  
  // 加载月学习数据
  const loadMonthlyData = async () => {
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await studyRecordService.getMonthlyStudyData()
      
      // 模拟月学习数据
      return [
        { week: '第1周', duration: 540, words: 120, exercises: 4 },
        { week: '第2周', duration: 630, words: 140, exercises: 5 },
        { week: '第3周', duration: 720, words: 160, exercises: 6 },
        { week: '第4周', duration: 810, words: 180, exercises: 7 }
      ]
    } catch (error) {
      console.error('加载月学习数据失败:', error)
      return []
    }
  }
  
  // 加载分类学习数据
  const loadCategoryData = async () => {
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await studyRecordService.getCategoryStudyData()
      
      // 模拟分类学习数据
      return [
        { category: '词语学习', value: 45, color: '#1890ff' },
        { category: '成语学习', value: 25, color: '#52c41a' },
        { category: '语法练习', value: 15, color: '#faad14' },
        { category: '阅读理解', value: 15, color: '#f5222d' }
      ]
    } catch (error) {
      console.error('加载分类学习数据失败:', error)
      return []
    }
  }
  
  // 加载成就数据
  const loadAchievementData = async () => {
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await studyRecordService.getUserAchievements()
      
      // 模拟成就数据
      return [
        {          id: 1,          name: '初次登录',          description: '首次登录系统',          date: '2025-08-01',          icon: '🏆',          completed: true        },        {          id: 2,          name: '学习达人',          description: '连续学习7天',          date: '2025-08-10',          icon: '🌟',          completed: true        },        {          id: 3,          name: '词语大师',          description: '学习词语超过200个',          date: '2025-08-15',          icon: '📚',          completed: true        },
        {
          id: 4,
          name: '成语专家',
          description: '学习成语超过100个',
          icon: '📖',
          completed: false,
          progress: 65
        },
        {
          id: 5,
          name: '百日学习',
          description: '累计学习100天',
          icon: '🔥',
          completed: false,
          progress: 45
        }
      ]
    } catch (error) {
      console.error('加载成就数据失败:', error)
      return []
    }
  }
  
  // 加载详细学习记录
  const loadDetailedRecords = async () => {
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await studyRecordService.getDetailedStudyRecords(reportType, timeRange)
      
      // 模拟详细学习记录
      return [
        { id: 1, date: '2025-08-10 14:30', type: '词语学习', content: '高中语文词语练习', duration: 30, score: 92, status: 'completed' },        { id: 2, date: '2025-08-09 16:45', type: '成语学习', content: '四字成语辨析', duration: 25, score: 88, status: 'completed' },        { id: 3, date: '2025-08-08 19:20', type: '语法练习', content: '句子成分分析', duration: 40, score: 78, status: 'completed' },        { id: 4, date: '2025-08-07 10:15', type: '词语学习', content: '近义词辨析', duration: 35, score: 90, status: 'completed' },        { id: 5, date: '2025-08-06 15:30', type: '阅读理解', content: '现代文阅读练习', duration: 50, score: 85, status: 'completed' },        { id: 6, date: '2025-08-05 14:10', type: '成语学习', content: '成语接龙游戏', duration: 20, score: 94, status: 'completed' },        { id: 7, date: '2025-08-04 18:45', type: '语法练习', content: '病句修改练习', duration: 30, score: 82, status: 'completed' }
      ]
    } catch (error) {
      console.error('加载详细学习记录失败:', error)
      return []
    }
  }
  
  // 处理时间范围变化
  const handleTimeRangeChange = (dates) => {
    setTimeRange(dates)
    setReportType('custom')
  }
  
  // 处理报告类型变化
  const handleReportTypeChange = (value) => {
    setReportType(value)
    if (value !== 'custom') {
      setTimeRange(null)
    }
  }
  
  // 导出报告
  const exportReport = () => {
    // 在实际环境中，这里会实现导出功能
    message.success('报告导出成功')
  }
  
  // 重置筛选条件
  const resetFilters = () => {
    setReportType('weekly')
    setTimeRange(null)
  }
  
  // 格式化时长显示
  const formatDuration = (seconds) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    if (hours > 0) {
      return `${hours}小时${minutes}分钟`
    } else {
      return `${minutes}分钟`
    }
  }
  
  // 渲染学习时长趋势图配置
  const durationLineConfig = {
    data: weeklyData,
    xField: 'day',
    yField: 'duration',
    seriesField: 'day',
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
          name: datum.day,
          value: `${datum.duration}分钟`,
        }
      },
    },
  }
  
  // 渲染学习得分趋势图配置
  const scoreLineConfig = {
    data: weeklyData,
    xField: 'day',
    yField: 'score',
    seriesField: 'day',
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
          name: datum.day,
          value: `${datum.score}分`,
        }
      },
    },
  }
  
  // 渲染月度柱状图配置
  const monthlyBarConfig = {
    data: monthlyData,
    xField: 'week',
    yField: 'duration',
    seriesField: 'week',
    yAxis: {
      label: {
        formatter: (v) => `${v}分钟`,
      },
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
  
  // 渲染学习内容分布图配置
  const categoryPieConfig = {
    data: categoryData,
    angleField: 'value',
    colorField: 'category',
    radius: 0.8,
    label: {
      type: 'inner',
      offset: '-30%',
      content: ({ percent }) => `${(percent * 100).toFixed(0)}%`,
      style: {
        fontSize: 14,
        textAlign: 'center',
      },
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
          value: `${datum.value}%`,
        }
      },
    },
  }
  
  // 详细记录表格列配置
  const recordColumns = [
    {
      title: '学习时间',
      dataIndex: 'date',
      key: 'date',
      sorter: (a, b) => new Date(a.date) - new Date(b.date),
    },
    {
      title: '学习类型',
      dataIndex: 'type',
      key: 'type',
      filters: [
        { text: '词语学习', value: '词语学习' },
        { text: '成语学习', value: '成语学习' },
        { text: '语法练习', value: '语法练习' },
        { text: '阅读理解', value: '阅读理解' },
      ],
      onFilter: (value, record) => record.type === value,
    },
    {
      title: '学习内容',
      dataIndex: 'content',
      key: 'content',
    },
    {
      title: '学习时长',
      dataIndex: 'duration',
      key: 'duration',
      render: (text) => `${text}分钟`,
      sorter: (a, b) => a.duration - b.duration,
    },
    {
      title: '得分',
      dataIndex: 'score',
      key: 'score',
      render: (text) => `${text}分`,
      sorter: (a, b) => a.score - b.score,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (text) => (
        <span style={{ color: text === 'completed' ? '#52c41a' : '#faad14' }}>
          {text === 'completed' ? '已完成' : '进行中'}
        </span>
      ),
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Tooltip title="查看详情">
          <Button type="link" size="small">
            查看
          </Button>
        </Tooltip>
      ),
    },
  ]
  
  return (
    <div className="study-report-container">
      {/* 页面标题和筛选区域 */}
      <div className="report-header">
        <h1>学习报告</h1>
        <div className="filter-controls">
          <div className="filter-item">
            <span style={{ marginRight: 8 }}>报告周期：</span>
            <Select 
              value={reportType} 
              onChange={handleReportTypeChange} 
              style={{ width: 120 }} 
            >
              <Option value="weekly">本周</Option>
              <Option value="monthly">本月</Option>
              <Option value="custom">自定义</Option>
            </Select>
          </div>
          
          {reportType === 'custom' && (
            <div className="filter-item">
              <RangePicker onChange={handleTimeRangeChange} />
            </div>
          )}
          
          <div className="filter-actions">
            <Button onClick={resetFilters} style={{ marginRight: 8 }}>
              重置
            </Button>
            <Button type="primary" icon={<DownloadOutlined />} onClick={exportReport}>
              导出报告
            </Button>
          </div>
        </div>
      </div>
      
      {/* 统计卡片区域 */}
      <Row gutter={16} className="stats-cards">
        <Col span={6}>
          <Card>
            <Statistic 
              title="总学习时长" 
              value={studyStats.totalDuration} 
              suffix="分钟" 
              valueStyle={{ color: '#1890ff' }}
              prefix={<BookOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic 
              title="学习词语数" 
              value={studyStats.totalWords} 
              suffix="个" 
              valueStyle={{ color: '#52c41a' }}
              prefix={<BookOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic 
              title="完成练习数" 
              value={studyStats.totalExercises} 
              suffix="次" 
              valueStyle={{ color: '#faad14' }}
              prefix={<TrophyOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic 
              title="平均得分" 
              value={studyStats.averageScore} 
              suffix="分" 
              precision={1}
              valueStyle={{ color: '#f5222d' }}
              prefix={<TrophyOutlined />}
            />
          </Card>
        </Col>
      </Row>
      
      <Divider />
      
      {/* 图表区域 */}
      <Tabs activeKey={activeTabKey} onChange={setActiveTabKey}>
        <Tabs.TabPane tab="学习趋势" key="1">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="每日学习时长" loading={loading}>
                {weeklyData.length > 0 ? (
                  <Line {...durationLineConfig} />
                ) : (
                  <Empty description="暂无数据" />
                )}
              </Card>
            </Col>
            <Col span={12}>
              <Card title="每日学习得分" loading={loading}>
                {weeklyData.length > 0 ? (
                  <Line {...scoreLineConfig} />
                ) : (
                  <Empty description="暂无数据" />
                )}
              </Card>
            </Col>
          </Row>
        </Tabs.TabPane>
        
        <Tabs.TabPane tab="月度概览" key="2">
          <Card title="月度学习时长统计" loading={loading}>
            {monthlyData.length > 0 ? (
              <Bar {...monthlyBarConfig} />
            ) : (
              <Empty description="暂无数据" />
            )}
          </Card>
        </Tabs.TabPane>
        
        <Tabs.TabPane tab="学习分布" key="3">
          <Card title="学习内容分布" loading={loading}>
            {categoryData.length > 0 ? (
              <div style={{ display: 'flex', justifyContent: 'center' }}>
                <div style={{ width: '50%' }}>
                  <Pie {...categoryPieConfig} />
                </div>
                <div style={{ width: '50%', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                  {categoryData.map((item) => (
                    <div key={item.category} style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
                      <div style={{ width: 12, height: 12, backgroundColor: item.color, marginRight: 8 }}></div>
                      <span style={{ flex: 1 }}>{item.category}</span>
                      <span>{item.value}%</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <Empty description="暂无数据" />
            )}
          </Card>
        </Tabs.TabPane>
        
        <Tabs.TabPane tab="成就进度" key="4">
          <Card title="学习成就" loading={loading}>
            <div className="achievements-list">
              {achievementData.length > 0 ? (
                achievementData.map((achievement) => (
                  <div key={achievement.id} className="achievement-item">
                    <div className="achievement-icon">{achievement.icon}</div>
                    <div className="achievement-info">
                      <div className="achievement-name">{achievement.name}</div>
                      <div className="achievement-description">{achievement.description}</div>
                      {achievement.completed && (
                        <div className="achievement-date">完成日期：{achievement.date}</div>
                      )}
                      {!achievement.completed && achievement.progress !== undefined && (
                        <div className="achievement-progress">
                          <div className="progress-bar">
                            <div 
                              className="progress-fill" 
                              style={{ width: `${achievement.progress}%` }}
                            ></div>
                          </div>
                          <div className="progress-text">{achievement.progress}%</div>
                        </div>
                      )}
                    </div>
                    <div className="achievement-status">
                      {achievement.completed ? (
                        <span style={{ color: '#52c41a' }}>已完成</span>
                      ) : (
                        <span style={{ color: '#faad14' }}>进行中</span>
                      )}
                    </div>
                  </div>
                ))
              ) : (
                <Empty description="暂无成就" />
              )}
            </div>
          </Card>
        </Tabs.TabPane>
      </Tabs>
      
      <Divider />
      
      {/* 详细记录表格 */}
      <Card title="学习记录详情" loading={loading}>
        {detailedRecords.length > 0 ? (
          <Table 
            columns={recordColumns} 
            dataSource={detailedRecords} 
            rowKey="id" 
            pagination={{ pageSize: 10 }}
          />
        ) : (
          <Empty description="暂无学习记录" />
        )}
      </Card>
    </div>
  )
}

export default StudyReport