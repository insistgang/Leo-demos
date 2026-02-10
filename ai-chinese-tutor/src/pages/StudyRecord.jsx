import React, { useState, useEffect } from 'react'
import { Card, Tabs, DatePicker, Select, Table, Statistic, Row, Col, Progress, Badge, Empty, message, Button } from 'antd'
import { BarChartOutlined, LineChartOutlined, PieChartOutlined, ArrowUpOutlined } from '@ant-design/icons'

const { TabPane } = Tabs
const { RangePicker } = DatePicker
const { Option } = Select

const StudyRecord = () => {
  // 状态管理
  const [currentTab, setCurrentTab] = useState('1')
  const [dateRange, setDateRange] = useState(null)
  const [recordType, setRecordType] = useState('all')
  const [records, setRecords] = useState([])
  const [loading, setLoading] = useState(false)
  const [statistics, setStatistics] = useState({
    totalStudyTime: 0,
    totalWordsLearned: 0,
    totalExercisesCompleted: 0,
    averageScore: 0
  })
  const [weeklyData, setWeeklyData] = useState([])
  const [monthlyData, setMonthlyData] = useState([])
  const [categoryData, setCategoryData] = useState([])
  const [streakData, setStreakData] = useState({ current: 0, max: 0 })
  
  // 表格列配置
  const tableColumns = [
    {
      title: '学习内容',
      dataIndex: 'title',
      key: 'title',
      sorter: (a, b) => a.title.localeCompare(b.title),
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: type => (
        <Badge color={getTypeColor(type)} text={type} />
      ),
      filters: [
        { text: '词语学习', value: '词语学习' },
        { text: '练习测试', value: '练习测试' },
        { text: '阅读理解', value: '阅读理解' },
        { text: '成语学习', value: '成语学习' },
      ],
      onFilter: (value, record) => record.type === value,
    },
    {
      title: '学习时长',
      dataIndex: 'duration',
      key: 'duration',
      render: duration => `${duration}分钟`,
      sorter: (a, b) => a.duration - b.duration,
    },
    {
      title: '完成度',
      dataIndex: 'completionRate',
      key: 'completionRate',
      render: rate => (
        <Progress percent={rate} size="small" status={rate === 100 ? "success" : "active"} />
      ),
      sorter: (a, b) => a.completionRate - b.completionRate,
    },
    {
      title: '得分',
      dataIndex: 'score',
      key: 'score',
      render: score => score ? `${score}分` : '-',
      sorter: (a, b) => (a.score || 0) - (b.score || 0),
    },
    {
      title: '学习日期',
      dataIndex: 'studyDate',
      key: 'studyDate',
      sorter: (a, b) => new Date(a.studyDate) - new Date(b.studyDate),
    },
  ]
  
  // 初始化数据
  useEffect(() => {
    loadAllData()
  }, [dateRange, recordType])
  
  // 加载所有数据
  const loadAllData = async () => {
    try {
      setLoading(true)
      
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 600))
      
      // 模拟学习记录数据
      const mockRecords = [
        {
          id: 1,
          title: '小学语文基础知识练习',
          type: '练习测试',
          duration: 12,
          completionRate: 35,
          score: 70,
          studyDate: '2023-06-15'
        },
        {
          id: 2,
          title: '成语填空专项训练',
          type: '成语学习',
          duration: 8,
          completionRate: 0,
          score: null,
          studyDate: '2023-06-15'
        },
        {
          id: 3,
          title: '常见错别字辨析',
          type: '词语学习',
          duration: 15,
          completionRate: 100,
          score: 92,
          studyDate: '2023-06-14'
        },
        {
          id: 4,
          title: '阅读理解提升练习',
          type: '阅读理解',
          duration: 25,
          completionRate: 60,
          score: 80,
          studyDate: '2023-06-10'
        },
        {
          id: 5,
          title: '古诗词默写与理解',
          type: '练习测试',
          duration: 22,
          completionRate: 100,
          score: 92,
          studyDate: '2023-06-12'
        },
        {
          id: 6,
          title: '病句修改练习',
          type: '练习测试',
          duration: 13,
          completionRate: 100,
          score: 85,
          studyDate: '2023-06-08'
        }
      ]
      
      // 模拟统计数据
      const mockStatistics = {
        totalStudyTime: 1800, // 分钟
        totalWordsLearned: 320,
        totalExercisesCompleted: 15,
        averageScore: 85
      }
      
      // 模拟周学习数据（最近7天）
      const mockWeeklyData = [
        { day: '周一', studyTime: 45, exercises: 2, wordsLearned: 15 },
        { day: '周二', studyTime: 60, exercises: 3, wordsLearned: 25 },
        { day: '周三', studyTime: 30, exercises: 1, wordsLearned: 10 },
        { day: '周四', studyTime: 90, exercises: 4, wordsLearned: 35 },
        { day: '周五', studyTime: 0, exercises: 0, wordsLearned: 0 },
        { day: '周六', studyTime: 75, exercises: 3, wordsLearned: 30 },
        { day: '周日', studyTime: 30, exercises: 1, wordsLearned: 10 }
      ]
      
      // 模拟月学习数据（最近6个月）
      const mockMonthlyData = [
        { month: '1月', studyTime: 320, exercises: 15, averageScore: 78 },
        { month: '2月', studyTime: 450, exercises: 20, averageScore: 80 },
        { month: '3月', studyTime: 380, exercises: 18, averageScore: 82 },
        { month: '4月', studyTime: 520, exercises: 25, averageScore: 85 },
        { month: '5月', studyTime: 480, exercises: 22, averageScore: 83 },
        { month: '6月', studyTime: 350, exercises: 16, averageScore: 86 }
      ]
      
      // 模拟分类学习数据
      const mockCategoryData = [
        { category: '词语学习', studyTime: 650, percentage: 35 },
        { category: '练习测试', studyTime: 720, percentage: 39 },
        { category: '阅读理解', studyTime: 320, percentage: 17 },
        { category: '成语学习', studyTime: 180, percentage: 9 }
      ]
      
      // 模拟连续学习数据
      const mockStreakData = {
        current: 3, // 当前连续学习天数
        max: 15     // 最大连续学习天数
      }
      
      // 根据筛选条件过滤记录数据
      let filteredRecords = [...mockRecords]
      
      if (recordType !== 'all') {
        filteredRecords = filteredRecords.filter(record => record.type === recordType)
      }
      
      if (dateRange) {
        const [startDate, endDate] = dateRange
        filteredRecords = filteredRecords.filter(record => {
          const recordDate = new Date(record.studyDate)
          return recordDate >= startDate && recordDate <= endDate
        })
      }
      
      // 更新状态
      setRecords(filteredRecords)
      setStatistics(mockStatistics)
      setWeeklyData(mockWeeklyData)
      setMonthlyData(mockMonthlyData)
      setCategoryData(mockCategoryData)
      setStreakData(mockStreakData)
    } catch (error) {
      message.error('加载数据失败')
      console.error('加载数据失败:', error)
    } finally {
      setLoading(false)
    }
  }
  
  // 根据类型获取颜色
  const getTypeColor = (type) => {
    const colorMap = {
      '词语学习': 'blue',
      '练习测试': 'green',
      '阅读理解': 'orange',
      '成语学习': 'red'
    }
    return colorMap[type] || 'purple'
  }
  
  // 根据索引获取颜色
  const getIndexColor = (index) => {
    const colors = ['#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1', '#13c2c2']
    return colors[index % colors.length]
  }
  
  // 处理日期范围变化
  const handleDateRangeChange = (dates, dateStrings) => {
    setDateRange(dates)
  }
  
  // 处理记录类型变化
  const handleRecordTypeChange = (value) => {
    setRecordType(value)
  }
  
  // 渲染学习统计卡片
  const renderStatisticCards = () => (
    <Row gutter={16} className="statistic-cards">
      <Col span={6}>
        <Card className="stat-card">
          <Statistic
            title="总学习时长"
            value={Math.floor(statistics.totalStudyTime / 60)}
            suffix="小时"
            precision={1}
            valueStyle={{ color: '#3f8600' }}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card className="stat-card">
          <Statistic
            title="累计学习词语"
            value={statistics.totalWordsLearned}
            suffix="个"
            precision={0}
            valueStyle={{ color: '#1890ff' }}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card className="stat-card">
          <Statistic
            title="完成练习数"
            value={statistics.totalExercisesCompleted}
            suffix="次"
            precision={0}
            valueStyle={{ color: '#fa8c16' }}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card className="stat-card">
          <Statistic
            title="平均得分"
            value={statistics.averageScore}
            suffix="分"
            precision={1}
            valueStyle={{ color: '#f5222d' }}
          />
        </Card>
      </Col>
    </Row>
  )
  
  // 渲染连续学习卡片
  const renderStreakCard = () => (
    <Card className="streak-card">
      <div className="streak-content">
        <ArrowUpOutlined className="streak-icon" />
        <div className="streak-info">
          <div className="streak-item">
            <span className="streak-label">当前连续学习：</span>
            <span className="streak-value current">{streakData.current}天</span>
          </div>
          <div className="streak-item">
            <span className="streak-label">最长连续学习：</span>
            <span className="streak-value max">{streakData.max}天</span>
          </div>
        </div>
      </div>
    </Card>
  )
  
  // 渲染周学习趋势
  const renderWeeklyChart = () => (
    <Card className="chart-card" title="周学习趋势" extra={<BarChartOutlined />}>
      <div className="weekly-chart">
        {weeklyData.map((item, index) => (
          <div key={item.day} className="chart-bar-wrapper">
            <div 
              className="chart-bar"
              style={{
                height: `${Math.max(5, (item.studyTime / Math.max(...weeklyData.map(d => d.studyTime || 1))) * 100)}%`,
                backgroundColor: '#1890ff'
              }}
            />
            <div className="chart-label">{item.day}</div>
            <div className="chart-value">{item.studyTime}m</div>
          </div>
        ))}
      </div>
      <div className="chart-legend">
        <div className="legend-item">
          <span className="legend-color" style={{ backgroundColor: '#1890ff' }}></span>
          <span>学习时长(分钟)</span>
        </div>
      </div>
    </Card>
  )
  
  // 渲染月学习趋势
  const renderMonthlyChart = () => (
    <Card className="chart-card" title="月学习趋势" extra={<LineChartOutlined />}>
      <div className="monthly-data-table">
        <div className="table-header">
          <div>月份</div>
          <div>学习时长(分钟)</div>
          <div>练习数</div>
          <div>平均得分</div>
        </div>
        {monthlyData.map(item => (
          <div key={item.month} className="table-row">
            <div>{item.month}</div>
            <div>{item.studyTime}</div>
            <div>{item.exercises}</div>
            <div>{item.averageScore}</div>
          </div>
        ))}
      </div>
    </Card>
  )
  
  // 渲染分类学习数据
  const renderCategoryChart = () => (
    <Card className="chart-card" title="学习内容分布" extra={<PieChartOutlined />}>
      <div className="category-list">
        {categoryData.map((item, index) => (
          <div key={item.category} className="category-item">
            <div className="category-info">
              <span className="category-name">{item.category}</span>
              <span className="category-percentage">{item.percentage}%</span>
            </div>
            <Progress 
              percent={item.percentage} 
              size="small" 
              strokeColor={getIndexColor(index)} 
            />
          </div>
        ))}
      </div>
    </Card>
  )
  
  // 渲染学习建议
  const renderSuggestions = () => (
    <Card className="analysis-card" title="学习建议">
      <div className="suggestions-list">
        <div className="suggestion-item">
          <Badge status="success" className="suggestion-badge" />
          <span>你的词语学习时间充足，继续保持！</span>
        </div>
        <div className="suggestion-item">
          <Badge status="warning" className="suggestion-badge" />
          <span>阅读理解练习相对较少，建议增加此类练习。</span>
        </div>
        <div className="suggestion-item">
          <Badge status="success" className="suggestion-badge" />
          <span>连续学习3天，坚持下去！</span>
        </div>
        <div className="suggestion-item">
          <Badge status="info" className="suggestion-badge" />
          <span>周五没有学习记录，建议保持每日学习习惯。</span>
        </div>
      </div>
    </Card>
  )
  
  // 渲染筛选器
  const renderFilters = () => (
    <div className="filters-container">
      <div className="filter-item">
        <span className="filter-label">日期范围：</span>
        <RangePicker onChange={handleDateRangeChange} className="date-picker" />
      </div>
      <div className="filter-item">
        <span className="filter-label">记录类型：</span>
        <Select 
          defaultValue="all" 
          style={{ width: 150 }} 
          onChange={handleRecordTypeChange}
          className="type-select"
        >
          <Option value="all">全部类型</Option>
          <Option value="词语学习">词语学习</Option>
          <Option value="练习测试">练习测试</Option>
          <Option value="阅读理解">阅读理解</Option>
          <Option value="成语学习">成语学习</Option>
        </Select>
      </div>
    </div>
  )
  
  return (
    <div className="study-record-container">
      <h2>学习记录</h2>
      
      {renderStatisticCards()}
      {renderStreakCard()}
      
      <Tabs activeKey={currentTab} onChange={setCurrentTab} className="record-tabs">
        <TabPane tab="学习记录" key="1">
          {renderFilters()}
          
          <Table
            columns={tableColumns}
            dataSource={records}
            rowKey="id"
            loading={loading}
            pagination={{ pageSize: 10 }}
            className="records-table"
            locale={{ emptyText: <Empty description="暂无学习记录" /> }}
          />
        </TabPane>
        
        <TabPane tab="学习分析" key="2">
          <Row gutter={16}>
            <Col span={12}>
              {renderWeeklyChart()}
            </Col>
            <Col span={12}>
              {renderMonthlyChart()}
            </Col>
            <Col span={12}>
              {renderCategoryChart()}
            </Col>
            <Col span={12}>
              {renderSuggestions()}
            </Col>
          </Row>
        </TabPane>
      </Tabs>
      
      {/* 添加开始学习按钮区域 */}
      <div className="start-learning-section">
        <h3>开始新的学习</h3>
        <Row gutter={16}>
          <Col span={8}>
            <Card className="start-learning-card">
              <h4>词语学习</h4>
              <p>学习新词语，提升词汇量</p>
              <Button type="primary" block>开始学习</Button>
            </Card>
          </Col>
          <Col span={8}>
            <Card className="start-learning-card">
              <h4>练习测试</h4>
              <p>进行语文知识练习</p>
              <Button type="primary" block>开始练习</Button>
            </Card>
          </Col>
          <Col span={8}>
            <Card className="start-learning-card">
              <h4>阅读理解</h4>
              <p>提升阅读理解能力</p>
              <Button type="primary" block>开始阅读</Button>
            </Card>
          </Col>
        </Row>
      </div>
    </div>
  )
}

export default StudyRecord