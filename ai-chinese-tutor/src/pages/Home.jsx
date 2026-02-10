import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Statistic, Button, Avatar, Badge, Tabs, Spin, Tag } from 'antd'
import { FileTextOutlined, AudioOutlined, HighlightOutlined, BookOutlined, LayoutOutlined, StarOutlined, TrophyOutlined, ClockCircleOutlined, ExclamationCircleOutlined, RobotOutlined, CalendarOutlined, EditOutlined } from '@ant-design/icons'
import { useNavigate, Link } from 'react-router-dom'
import { authService } from '../services/authService'
import { studyRecordService } from '../services/studyRecordService'
import api from '../services/api'

const { TabPane } = Tabs

const Home = () => {
  const [userInfo, setUserInfo] = useState(null)
  const [loading, setLoading] = useState(true)
  const [statsLoading, setStatsLoading] = useState(true)
  const [currentTime, setCurrentTime] = useState(new Date())

  const [studyStats, setStudyStats] = useState({
    todayScore: 0,
    errorCount: 0,
    totalStudyTime: 0,
    studyDays: 0
  })

  const [questionStats, setQuestionStats] = useState({})

  const [recommendList, setRecommendList] = useState([
    {
      id: 1,
      title: '汉字拼音练习',
      description: '练习常用汉字的拼音读音',
      type: 'pinyin',
      isNew: false,
      icon: <AudioOutlined />,
      path: '/pinyin'
    },
    {
      id: 2,
      title: '错别字纠错',
      description: '智能识别并纠正错别字',
      type: 'correction',
      isNew: true,
      icon: <HighlightOutlined />,
      path: '/correction'
    },
    {
      id: 3,
      title: '古诗词赏析',
      description: '学习经典古诗词',
      type: 'poetry',
      isNew: false,
      icon: <BookOutlined />,
      path: '/literature'
    }
  ])

  // 定义所有模块的配置
  const allModulesConfig = {
    pinyin: {
      title: '拼音学习',
      description: '汉字转拼音\n语音评测',
      icon: <AudioOutlined />,
      path: '/pinyin'
    },
    vocabulary: {
      title: '词汇学习',
      description: '词语查询\n语义理解',
      icon: <FileTextOutlined />,
      path: '/vocabulary'
    },
    literature: {
      title: '古诗词',
      description: '古诗词学习\n名句默写',
      icon: <BookOutlined />,
      path: '/literature'
    },
    idiom: {
      title: '成语熟语',
      description: '成语学习\n谚语理解',
      icon: <StarOutlined />,
      path: '/idiom'
    },
    correction: {
      title: '病句修改',
      description: '病句辨析\n智能纠错',
      icon: <HighlightOutlined />,
      path: '/correction'
    },
    grammar: {
      title: '语法知识',
      description: '语法练习\n句子分析',
      icon: <LayoutOutlined />,
      path: '/exercise'
    },
    reading: {
      title: '阅读理解',
      description: '文章阅读\n理解分析',
      icon: <FileTextOutlined />,
      path: '/exercise'
    },
    comprehension: {
      title: '综合练习',
      description: '综合能力\n全面测试',
      icon: <LayoutOutlined />,
      path: '/exercise'
    }
  }

  const [moduleList, setModuleList] = useState([])

  // 获取用户信息
  useEffect(() => {
    const fetchUserInfo = async () => {
      try {
        const user = authService.getStoredUser();
        if (user) {
          setUserInfo({
            name: user.nickname || user.username,
            level: '初级',
            avatar: user.avatar || `https://api.dicebear.com/7.x/avataaars/svg?seed=${user.username}`
          });
        }
      } catch (error) {
        console.error('获取用户信息失败:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchUserInfo();
  }, []);

  // 更新当前时间
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  // 获取学习统计数据
  useEffect(() => {
    const fetchStatistics = async () => {
      try {
        console.log('========== Home.jsx - 开始获取统计数据 ==========');
        setStatsLoading(true);
        const stats = await studyRecordService.getHomeStatistics();
        console.log('Home.jsx - 获取到的统计数据:', stats);
        setStudyStats({
          todayScore: stats.todayScore,
          errorCount: stats.errorCount,
          totalStudyTime: stats.totalStudyTime,
          studyDays: stats.studyDays
        });
        console.log('Home.jsx - 状态已更新, studyStats:', {
          todayScore: stats.todayScore,
          errorCount: stats.errorCount,
          totalStudyTime: stats.totalStudyTime,
          studyDays: stats.studyDays
        });
      } catch (error) {
        console.error('获取统计数据失败:', error);
      } finally {
        setStatsLoading(false);
      }
    };

    fetchStatistics();
  }, []);

  // 获取题目统计
  useEffect(() => {
    const fetchQuestionStats = async () => {
      try {
        const response = await api.get('/questions/stats');
        if (response.success) {
          setQuestionStats(response.data);

          // 动态生成模块列表
          const modules = Object.entries(response.data)
            .filter(([type, count]) => count > 0 && allModulesConfig[type])
            .map(([type, count]) => ({
              id: type,
              title: allModulesConfig[type].title,
              description: allModulesConfig[type].description,
              icon: allModulesConfig[type].icon,
              path: allModulesConfig[type].path,
              type: type
            }));
          setModuleList(modules);
        }
      } catch (error) {
        console.error('获取题目统计失败:', error);
      }
    };

    fetchQuestionStats();
  }, []);

  // 格式化学习时间
  const formatStudyTime = (minutes) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    if (hours > 0) {
      return `${hours}小时${mins > 0 ? mins + '分钟' : ''}`;
    }
    return `${mins}分钟`;
  };

  return (
    <div className="home-container">
      {/* 第一行：用户信息卡片和日期时间卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={24} md={12}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <Avatar
                size={80}
                src={userInfo?.avatar}
                style={{ marginBottom: 16 }}
              >
                {userInfo?.name?.charAt(0)}
              </Avatar>
              <h3>{userInfo?.name || '加载中...'}</h3>
              <Badge
                count={userInfo?.level || '初级'}
                style={{ backgroundColor: '#52c41a' }}
              />
              <p style={{ marginTop: 8, color: '#666' }}>
                {statsLoading ? (
                  <Spin size="small" />
                ) : (
                  `已坚持学习 ${studyStats.studyDays} 天`
                )}
              </p>
            </div>
          </Card>
        </Col>

        {/* 日期时间卡片 */}
        <Col xs={24} sm={24} md={12}>
          <Card>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 48, color: '#1890ff', marginBottom: 16 }}>
                <CalendarOutlined />
              </div>
              <div style={{ fontSize: 24, fontWeight: 'bold', marginBottom: 8 }}>
                {currentTime.toLocaleDateString('zh-CN', {
                  month: 'long',
                  day: 'numeric'
                })}
              </div>
              <div style={{ fontSize: 16, color: '#666', marginBottom: 8 }}>
                {currentTime.toLocaleDateString('zh-CN', {
                  weekday: 'long'
                })}
              </div>
              <div style={{ fontSize: 14, color: '#666', marginBottom: 4 }}>
                {currentTime.getFullYear()}年
              </div>
              <div style={{ fontSize: 20, fontWeight: 'bold', color: '#1890ff' }}>
                {currentTime.toLocaleTimeString('zh-CN', {
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit'
                })}
              </div>
            </div>
          </Card>
        </Col>
      </Row>

      {/* 第二行：学习统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={24} md={24}>
          <Row gutter={[16, 16]}>
            <Col xs={12} sm={6}>
              <Card>
                <Statistic
                  title="今日得分"
                  value={studyStats.todayScore}
                  suffix="分"
                  valueStyle={{ color: '#3f8600' }}
                  prefix={<TrophyOutlined />}
                  loading={statsLoading}
                />
              </Card>
            </Col>
            <Col xs={12} sm={6}>
              <Card>
                <Statistic
                  title="总学习时间"
                  value={formatStudyTime(studyStats.totalStudyTime)}
                  valueStyle={{ color: '#1890ff' }}
                  prefix={<ClockCircleOutlined />}
                  loading={statsLoading}
                />
              </Card>
            </Col>
            <Col xs={12} sm={6}>
              <Card>
                <Statistic
                  title="已学习天数"
                  value={studyStats.studyDays}
                  suffix="天"
                  valueStyle={{ color: '#722ed1' }}
                  prefix={<CalendarOutlined />}
                  loading={statsLoading}
                />
              </Card>
            </Col>
            <Col xs={12} sm={6}>
              <Card>
                <Statistic
                  title="错题数量"
                  value={studyStats.errorCount}
                  suffix="个"
                  valueStyle={{ color: '#cf1322' }}
                  prefix={<ExclamationCircleOutlined />}
                  loading={statsLoading}
                />
              </Card>
            </Col>
          </Row>
        </Col>
      </Row>

      {/* 推荐学习 */}
      <Card
        title={
          <span>
            推荐学习
            <span style={{ fontSize: 14, fontWeight: 'normal', marginLeft: 10, color: '#666' }}>
              {new Date().toLocaleDateString('zh-CN', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                weekday: 'long'
              })}
            </span>
          </span>
        }
        style={{ marginBottom: 24 }}
      >
        <Row gutter={[16, 16]}>
          {recommendList.map((item) => (
            <Col xs={24} sm={8} key={item.id}>
              <Card
                hoverable
                style={{ textAlign: 'center', height: '100%' }}
                bodyStyle={{ padding: 24 }}
              >
                <div style={{ fontSize: 32, color: '#1890ff', marginBottom: 16 }}>
                  {item.icon}
                </div>
                <h4>{item.title}</h4>
                <p style={{ color: '#666', marginBottom: 16 }}>{item.description}</p>
                {item.isNew && (
                  <Badge
                    count="新"
                    style={{ backgroundColor: '#52c41a', marginBottom: 16 }}
                  />
                )}
                <br />
                <Button
                  type="primary"
                  icon={<RobotOutlined />}
                  onClick={() => window.location.href = item.path}
                >
                  开始学习
                </Button>
              </Card>
            </Col>
          ))}
        </Row>
      </Card>

      {/* 模拟考试 */}
      <Card title="模拟考试" style={{ marginBottom: 24 }}>
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} md={8}>
            <Card
              hoverable
              style={{ textAlign: 'center', height: '100%' }}
              bodyStyle={{ padding: 24 }}
              onClick={() => window.location.href = '/mock-exam'}
            >
              <div style={{ fontSize: 48, color: '#fa8c16', marginBottom: 16 }}>
                <TrophyOutlined />
              </div>
              <h3>开始模拟考试</h3>
              <p style={{ color: '#666' }}>
                检验学习成果<br />
                随机抽取题目组卷<br />
                支持多种题型选择
              </p>
              <Button type="primary" size="large" icon={<EditOutlined />}>
                进入考场
              </Button>
            </Card>
          </Col>
          <Col xs={24} sm={12} md={16}>
            <div style={{ padding: '20px 0' }}>
              <h4>模拟考试说明</h4>
              <ul style={{ lineHeight: '1.8' }}>
                <li>可自定义考试时长：30/60/90/120分钟</li>
                <li>可自定义题目数量：20/50/100题</li>
                <li>可按题型筛选：拼音、词汇、古诗词等</li>
                <li>考试结束自动评分，查看详细解析</li>
                <li>考试成绩自动保存到学习记录</li>
              </ul>
              <div style={{ marginTop: 20 }}>
                <Tag color="blue">限时考试</Tag>
                <Tag color="green">自动评分</Tag>
                <Tag color="orange">详细解析</Tag>
              </div>
            </div>
          </Col>
        </Row>
      </Card>

      {/* 学习模块 */}
      <Card title="学习模块">
        <Row gutter={[16, 16]}>
          {moduleList.map((item) => (
            <Col xs={24} sm={8} md={4} key={item.id}>
              <Card
                hoverable
                style={{ textAlign: 'center', height: '100%' }}
                bodyStyle={{ padding: 16 }}
                onClick={() => window.location.href = item.path}
              >
                <div style={{ fontSize: 28, color: '#1890ff', marginBottom: 8 }}>
                  {item.icon}
                </div>
                <h5>{item.title}</h5>
                <p style={{ fontSize: 12, color: '#666', whiteSpace: 'pre-line' }}>
                  {item.description}
                </p>
                <Badge count={questionStats[item.type]} style={{ backgroundColor: '#52c41a' }} />
              </Card>
            </Col>
          ))}
        </Row>
      </Card>
    </div>
  )
}

export default Home