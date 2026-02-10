import React, { useState } from 'react'
import { Card, Row, Col, Button, Tag, Typography, Divider, Descriptions, Badge, Space } from 'antd'
import { StarOutlined, MessageOutlined, HomeOutlined, FileTextOutlined, BarChartOutlined, UserOutlined } from '@ant-design/icons'
import { useNavigate, useParams } from 'react-router-dom'

const { Title, Paragraph, Text } = Typography

const ExerciseDetail = () => {
  const navigate = useNavigate()
  const { id } = useParams()
  const [isFavorite, setIsFavorite] = useState(false)
  const [liked, setLiked] = useState(false)
  const [likesCount, setLikesCount] = useState(128)

  // 模拟练习详情数据
  const exerciseDetail = {
    id: id || '1',
    title: '下列句子中没有错别字的一项是',
    type: '选择题',
    difficulty: '中等',
    subject: '语文基础',
    category: '错别字纠错',
    createdAt: '2025-08-10',
    question: '下列句子中没有错别字的一项是：',
    options: [
      {
        id: 'A',
        text: '我们要发扬团队精神，众志成诚，完成这项艰巨的任务。',
        isCorrect: false,
        explanation: '错误："众志成诚"应为"众志成城"'
      },
      {
        id: 'B',
        text: '阅读经典文学作品，可以潜移默化地提升我们的文化素养。',
        isCorrect: true,
        explanation: '正确'
      },
      {
        id: 'C',
        text: '他废尽心思，终于找到了问题的解决方法。',
        isCorrect: false,
        explanation: '错误："废尽心思"应为"费尽心思"'
      },
      {
        id: 'D',
        text: '老师的谆谆教悔，我将铭记于心。',
        isCorrect: false,
        explanation: '错误："教悔"应为"教诲"'
      }
    ],
    analysis: '这道题主要考查学生对常见错别字的辨析能力。选项B中的句子没有错别字。选项A中"众志成诚"的"诚"应改为"城"，"众志成城"比喻大家团结一致，就能克服困难；选项C中"废尽心思"的"废"应改为"费"，"费尽心思"指想尽一切办法；选项D中"教悔"的"悔"应改为"诲"，"教诲"指教导训诫。',
    knowledgePoint: '错别字辨析是语文基础的重要内容，需要学生平时多积累，注意形似字、同音字的区别。',
    userAnswer: 'B', // 模拟用户答案
    isCorrect: true,
    answerTime: '2025-08-15 14:30:25',
    spendTime: 45, // 单位：秒
    relatedExercises: [
      {
        id: '2',
        title: '下列成语中没有错别字的一组是',
        type: '选择题',
        difficulty: '中等'
      },
      {
        id: '3',
        title: '找出并改正下列句子中的错别字',
        type: '改错题',
        difficulty: '较难'
      },
      {
        id: '4',
        title: '下列词语中书写全部正确的一项是',
        type: '选择题',
        difficulty: '简单'
      }
    ]
  }

  const handleFavorite = () => {
    setIsFavorite(!isFavorite)
  }

  const handleLike = () => {
    if (liked) {
      setLikesCount(likesCount - 1)
    } else {
      setLikesCount(likesCount + 1)
    }
    setLiked(!liked)
  }

  const handleShare = () => {
    // 分享功能逻辑
    console.log('分享练习', exerciseDetail.id)
  }

  const handleBack = () => {
    navigate(-1)
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}分${secs}秒`
  }

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case '简单':
        return 'green'
      case '中等':
        return 'blue'
      case '较难':
        return 'orange'
      case '困难':
        return 'red'
      default:
        return 'blue'
    }
  }

  return (
    <div className="exercise-detail-container">
      {/* 页面标题和操作栏 */}
      <div className="exercise-detail-header">
        <div className="header-actions">
          <Button type="default" onClick={handleBack}>
            返回
          </Button>
          <Space style={{ marginLeft: 'auto' }}>
            <Button 
              type="text" 
              icon={isFavorite ? <StarOutlined style={{ color: '#ffd700' }} /> : <StarOutlined />}
              onClick={handleFavorite}
            >
              {isFavorite ? '已收藏' : '收藏'}
            </Button>
            <Button type="default" onClick={handleShare}>
              分享
            </Button>
          </Space>
        </div>
      </div>

      {/* 练习详情卡片 */}
      <Card className="exercise-detail-card">
        <div className="exercise-detail-meta">
          <div className="exercise-detail-tags">
            <Tag color={getDifficultyColor(exerciseDetail.difficulty)}>{exerciseDetail.difficulty}</Tag>
            <Tag>{exerciseDetail.type}</Tag>
            <Tag>{exerciseDetail.category}</Tag>
            <Tag>{exerciseDetail.subject}</Tag>
          </div>
          <div className="exercise-detail-info">
            <Text type="secondary">发布于：{exerciseDetail.createdAt}</Text>
          </div>
        </div>

        <Divider />

        {/* 题目内容 */}
        <div className="exercise-detail-content">
          <Title level={4} className="exercise-question">
            {exerciseDetail.question}
          </Title>

          {/* 选项 */}
          <div className="exercise-options">
            {exerciseDetail.options.map((option) => {
              let optionClass = 'option-btn'
              if (exerciseDetail.userAnswer === option.id) {
                optionClass += option.isCorrect ? ' correct-option' : ' incorrect-option'
              } else if (option.isCorrect) {
                optionClass += ' correct-option'
              }
              
              return (
                <div key={option.id} className={optionClass}>
                  <div className="option-header">
                    <span className="option-label">{option.id}.</span>
                    <span className="option-text">{option.text}</span>
                    {exerciseDetail.userAnswer === option.id && (
                      <Badge 
                        status={option.isCorrect ? "success" : "error"} 
                        text={option.isCorrect ? "正确" : "错误"} 
                        style={{ marginLeft: 8 }}
                      />
                    )}
                  </div>
                  {exerciseDetail.userAnswer && (
                    <div className="option-explanation">
                      <Text type="secondary">{option.explanation}</Text>
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          <Divider />

          {/* 解析 */}
          <div className="exercise-analysis">
            <Title level={5}>题目解析</Title>
            <Paragraph>{exerciseDetail.analysis}</Paragraph>
          </div>

          {/* 知识点 */}
          <div className="exercise-knowledge-point">
            <Title level={5}>知识点</Title>
            <Paragraph>{exerciseDetail.knowledgePoint}</Paragraph>
          </div>

          <Divider />

          {/* 用户答题情况 */}
          <div className="exercise-user-status">
            <Descriptions column={2} bordered>
              <Descriptions.Item label="你的答案">
                <Text strong>{exerciseDetail.userAnswer}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="答题结果">
                <Badge 
                  status={exerciseDetail.isCorrect ? "success" : "error"} 
                  text={exerciseDetail.isCorrect ? "正确" : "错误"} 
                />
              </Descriptions.Item>
              <Descriptions.Item label="答题时间">
                {exerciseDetail.answerTime}
              </Descriptions.Item>
              <Descriptions.Item label="用时">
                {formatTime(exerciseDetail.spendTime)}
              </Descriptions.Item>
            </Descriptions>
          </div>

          {/* 互动区域 */}
          <div className="exercise-interaction">
            <Button 
              type="text" 
              onClick={handleLike}
              className={liked ? 'liked' : ''}
            >
              {liked ? '已赞' : '点赞'} ({likesCount})
            </Button>
            <Button type="text">
              评论
            </Button>
          </div>
        </div>
      </Card>

      {/* 相关练习推荐 */}
      <Card title="相关练习推荐" className="related-exercises-card" style={{ marginTop: 24 }}>
        <div className="related-exercises-list">
          {exerciseDetail.relatedExercises.map((exercise) => (
            <Card 
              key={exercise.id} 
              hoverable 
              className="related-exercise-item"
              onClick={() => navigate(`/exercise-detail/${exercise.id}`)}
            >
              <div className="related-exercise-content">
                <h4 className="related-exercise-title">{exercise.title}</h4>
                <div className="related-exercise-meta">
                  <Tag>{exercise.type}</Tag>
                  <Tag color={getDifficultyColor(exercise.difficulty)}>{exercise.difficulty}</Tag>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </Card>

      {/* 底部导航 */}
      <div className="exercise-detail-footer">
        <Button type="primary" className="continue-btn" onClick={() => navigate('/exercise')}>
          继续练习
        </Button>
        <div className="quick-nav">
          <Button type="text" icon={<HomeOutlined />} onClick={() => navigate('/')}>
            首页
          </Button>
          <Button type="text" icon={<FileTextOutlined />} onClick={() => navigate('/exercise')}>
            练习中心
          </Button>
          <Button type="text" icon={<BarChartOutlined />} onClick={() => navigate('/study-report')}>
            学习报告
          </Button>
          <Button type="text" icon={<UserOutlined />} onClick={() => navigate('/profile')}>
            个人中心
          </Button>
        </div>
      </div>
    </div>
  )
}

export default ExerciseDetail