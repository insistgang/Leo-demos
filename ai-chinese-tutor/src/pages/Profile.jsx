import React, { useState, useEffect } from 'react'
import { Card, Form, Input, Button, Avatar, Upload, Divider, message, List, Badge, Tag, Modal, Select, Switch, Row, Col } from 'antd'
import { UserOutlined, UploadOutlined, EditOutlined, SaveOutlined, EyeOutlined, EyeInvisibleOutlined, LockOutlined, BookOutlined, TrophyOutlined, CalendarOutlined, PhoneOutlined, MailOutlined, HomeOutlined, SettingOutlined } from '@ant-design/icons'
import { userService } from '../services/userService'

const { Option } = Select
const { TextArea } = Input

const Profile = () => {
  // 状态管理
  const [userInfo, setUserInfo] = useState({
    name: '',
    avatar: '',
    nickname: '',
    email: '',
    phone: '',
    gender: '',
    birthday: '',
    school: '',
    grade: '',
    interests: [],
    bio: ''
  })
  const [isEditing, setIsEditing] = useState(false)
  const [form] = Form.useForm()
  const [uploading, setUploading] = useState(false)
  const [avatarUrl, setAvatarUrl] = useState('')
  const [showPasswordModal, setShowPasswordModal] = useState(false)
  const [passwordForm] = Form.useForm()
  const [userStats, setUserStats] = useState({
    wordsLearned: 0,
    exercisesCompleted: 0,
    studyDays: 0,
    achievementCount: 0
  })
  const [achievements, setAchievements] = useState([])
  const [settings, setSettings] = useState({
    dailyReminder: true,
    notificationEnabled: true,
    darkMode: false,
    language: 'zh-CN'
  })
  
  // 初始化数据
  useEffect(() => {
    loadUserInfo()
    loadUserStats()
    loadAchievements()
    loadSettings()
  }, [])
  
  // 加载用户信息
  const loadUserInfo = async () => {
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 400))
      
      // 模拟用户信息数据
      const mockUserInfo = {
        id: '1001',
        name: '张三',
        avatar: 'https://api.dicebear.com/6.x/avataaars/svg?seed=张三',
        nickname: '体育小王子',
        email: 'zhangsan@example.com',
        phone: '13800138000',
        gender: 'male',
        birthday: '2005-06-15',
        school: '某某高中',
        grade: '高二',
        interests: ['语文', '体育', '历史'],
        bio: '我是一名高二的体育生，热爱篮球和跑步，正在努力提高语文成绩！'
      }
      
      setUserInfo(mockUserInfo)
      setAvatarUrl(mockUserInfo.avatar)
      form.setFieldsValue(mockUserInfo)
    } catch (error) {
      message.error('加载用户信息失败')
    }
  }
  
  // 加载用户统计数据
  const loadUserStats = async () => {
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 300))
      
      // 模拟用户统计数据
      const mockUserStats = {
        wordsLearned: 320,
        exercisesCompleted: 15,
        studyDays: 45,
        achievementCount: 8
      }
      
      setUserStats(mockUserStats)
    } catch (error) {
      message.error('加载用户统计失败')
    }
  }
  
  // 加载成就数据
  const loadAchievements = async () => {
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 350))
      
      // 模拟成就数据
      const mockAchievements = [
        {
          id: 1,
          name: '初次登录',
          description: '首次登录系统',
          icon: '🏆',
          date: '2023-05-01',
          level: 'bronze'
        },
        {
          id: 2,
          name: '学习达人',
          description: '连续学习7天',
          icon: '🌟',
          date: '2023-05-10',
          level: 'silver'
        },
        {
          id: 3,
          name: '词语大师',
          description: '学习词语超过200个',
          icon: '📚',
          date: '2023-05-15',
          level: 'silver'
        },
        {
          id: 4,
          name: '练习能手',
          description: '完成练习超过10次',
          icon: '✏️',
          date: '2023-05-20',
          level: 'bronze'
        },
        {
          id: 5,
          name: '高分学霸',
          description: '单次练习得分超过95分',
          icon: '💯',
          date: '2023-05-25',
          level: 'gold'
        }
      ]
      
      setAchievements(mockAchievements)
    } catch (error) {
      message.error('加载成就数据失败')
    }
  }
  
  // 加载用户设置
  const loadSettings = async () => {
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 250))
      
      // 模拟设置数据
      const mockSettings = {
        dailyReminder: true,
        notificationEnabled: true,
        darkMode: false,
        language: 'zh-CN'
      }
      
      setSettings(mockSettings)
    } catch (error) {
      message.error('加载用户设置失败')
    }
  }
  
  // 处理头像上传
  const handleAvatarUpload = ({ file, onSuccess }) => {
    setUploading(true)
    // 模拟上传过程
    setTimeout(() => {
      // 模拟上传成功，使用随机图片URL
      const url = `https://api.dicebear.com/6.x/avataaars/svg?seed=${Math.random()}`
      setAvatarUrl(url)
      setUploading(false)
      onSuccess?.(file)
      message.success('头像上传成功')
    }, 1000)
  }
  
  // 开始编辑个人信息
  const startEditing = () => {
    setIsEditing(true)
  }
  
  // 保存个人信息
  const saveUserInfo = async (values) => {
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 500))
      
      const updatedUserInfo = {
        ...userInfo,
        ...values,
        avatar: avatarUrl
      }
      
      setUserInfo(updatedUserInfo)
      setIsEditing(false)
      message.success('个人信息保存成功')
    } catch (error) {
      message.error('保存个人信息失败')
    }
  }
  
  // 取消编辑
  const cancelEditing = () => {
    setIsEditing(false)
    form.setFieldsValue(userInfo)
    setAvatarUrl(userInfo.avatar)
  }
  
  // 打开修改密码弹窗
  const openPasswordModal = () => {
    setShowPasswordModal(true)
  }
  
  // 关闭修改密码弹窗
  const closePasswordModal = () => {
    setShowPasswordModal(false)
    passwordForm.resetFields()
  }
  
  // 修改密码
  const changePassword = async (values) => {
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 500))
      
      // 简单验证
      if (values.newPassword !== values.confirmPassword) {
        message.error('两次输入的密码不一致')
        return
      }
      
      closePasswordModal()
      message.success('密码修改成功')
    } catch (error) {
      message.error('修改密码失败')
    }
  }
  
  // 更新设置
  const updateSetting = async (key, value) => {
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 300))
      
      const updatedSettings = {
        ...settings,
        [key]: value
      }
      
      setSettings(updatedSettings)
      message.success('设置已更新')
    } catch (error) {
      message.error('更新设置失败')
    }
  }
  
  // 获取成就等级对应的颜色
  const getAchievementColor = (level) => {
    const colorMap = {
      bronze: '#cd7f32',
      silver: '#c0c0c0',
      gold: '#ffd700'
    }
    return colorMap[level] || '#888888'
  }
  
  // 获取成就等级对应的标签
  const getAchievementLevelTag = (level) => {
    const tagMap = {
      bronze: '铜',
      silver: '银',
      gold: '金'
    }
    return tagMap[level] || '普通'
  }
  
  return (
    <div className="profile-container">
      {/* 个人信息卡片 */}
      <Card className="profile-card">
        <div className="profile-header">
          <div className="avatar-section">
            <Avatar 
              size={120} 
              src={avatarUrl} 
              icon={<UserOutlined />}
              className="user-avatar"
            />
            {isEditing && (
              <Upload.Dragger
                accept="image/*"
                beforeUpload={() => false}
                customRequest={handleAvatarUpload}
                showUploadList={false}
                className="avatar-uploader"
              >
                <Button icon={<UploadOutlined />} loading={uploading}>
                  更换头像
                </Button>
              </Upload.Dragger>
            )}
          </div>
          
          <div className="user-info-summary">
            {isEditing ? (
              <Form form={form} layout="vertical" onFinish={saveUserInfo}>
                <Form.Item label="用户名" name="name" rules={[{ required: true, message: '请输入用户名' }]}>
                  <Input placeholder="请输入用户名" />
                </Form.Item>
                <Form.Item label="昵称" name="nickname">
                  <Input placeholder="请输入昵称" />
                </Form.Item>
              </Form>
            ) : (
              <>
                <h2>{userInfo.name}</h2>
                {userInfo.nickname && <p className="nickname">{userInfo.nickname}</p>}
              </>
            )}
            
            <div className="profile-actions">
              {isEditing ? (
                <>
                  <Button type="primary" onClick={() => form.submit()} icon={<SaveOutlined />}>
                    保存
                  </Button>
                  <Button onClick={cancelEditing} style={{ marginLeft: 8 }}>
                    取消
                  </Button>
                </>
              ) : (
                <Button type="primary" onClick={startEditing} icon={<EditOutlined />}>
                  编辑资料
                </Button>
              )}
            </div>
          </div>
        </div>
        
        <Divider />
        
        {/* 详细信息 */}
        {isEditing ? (
          <Form form={form} layout="vertical" onFinish={saveUserInfo} className="detailed-info-form">
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item label="邮箱" name="email" rules={[{ type: 'email', message: '请输入有效的邮箱地址' }]}>
                  <Input placeholder="请输入邮箱" prefix={<MailOutlined />} />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label="手机号" name="phone" rules={[{ pattern: /^1[3-9]\d{9}$/, message: '请输入有效的手机号' }]}>
                  <Input placeholder="请输入手机号" prefix={<PhoneOutlined />} />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label="性别" name="gender">
                  <Select placeholder="请选择性别">
                    <Option value="male">男</Option>
                    <Option value="female">女</Option>
                    <Option value="other">其他</Option>
                  </Select>
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label="生日" name="birthday">
                  <Input placeholder="请输入生日" prefix={<CalendarOutlined />} />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label="学校" name="school">
                  <Input placeholder="请输入学校" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label="年级" name="grade">
                  <Input placeholder="请输入年级" />
                </Form.Item>
              </Col>
              <Col span={24}>
                <Form.Item label="兴趣爱好" name="interests">
                  <Select mode="tags" placeholder="请输入兴趣爱好（可多选）">
                    <Option value="语文">语文</Option>
                    <Option value="数学">数学</Option>
                    <Option value="英语">英语</Option>
                    <Option value="体育">体育</Option>
                    <Option value="音乐">音乐</Option>
                    <Option value="美术">美术</Option>
                    <Option value="历史">历史</Option>
                    <Option value="地理">地理</Option>
                    <Option value="生物">生物</Option>
                    <Option value="化学">化学</Option>
                  </Select>
                </Form.Item>
              </Col>
              <Col span={24}>
                <Form.Item label="个人简介" name="bio">
                  <TextArea rows={4} placeholder="请输入个人简介" />
                </Form.Item>
              </Col>
            </Row>
          </Form>
        ) : (
          <div className="detailed-info">
            <Row gutter={16}>
              <Col span={12}>
                <div className="info-item">
                  <span className="info-label">邮箱：</span>
                  <span className="info-value">{userInfo.email || '-'}</span>
                </div>
              </Col>
              <Col span={12}>
                <div className="info-item">
                  <span className="info-label">手机号：</span>
                  <span className="info-value">{userInfo.phone ? userInfo.phone.replace(/(\d{3})\d{4}(\d{4})/, '$1****$2') : '-'}</span>
                </div>
              </Col>
              <Col span={12}>
                <div className="info-item">
                  <span className="info-label">性别：</span>
                  <span className="info-value">{userInfo.gender === 'male' ? '男' : userInfo.gender === 'female' ? '女' : userInfo.gender || '-'}</span>
                </div>
              </Col>
              <Col span={12}>
                <div className="info-item">
                  <span className="info-label">生日：</span>
                  <span className="info-value">{userInfo.birthday || '-'}</span>
                </div>
              </Col>
              <Col span={12}>
                <div className="info-item">
                  <span className="info-label">学校：</span>
                  <span className="info-value">{userInfo.school || '-'}</span>
                </div>
              </Col>
              <Col span={12}>
                <div className="info-item">
                  <span className="info-label">年级：</span>
                  <span className="info-value">{userInfo.grade || '-'}</span>
                </div>
              </Col>
              <Col span={24}>
                <div className="info-item">
                  <span className="info-label">兴趣爱好：</span>
                  <div className="interests-tags">
                    {userInfo.interests && userInfo.interests.length > 0 ? (
                      userInfo.interests.map((interest, index) => (
                        <Tag key={index}>{interest}</Tag>
                      ))
                    ) : (
                      '-' 
                    )}
                  </div>
                </div>
              </Col>
              <Col span={24}>
                <div className="info-item">
                  <span className="info-label">个人简介：</span>
                  <p className="bio-text">{userInfo.bio || '-'}</p>
                </div>
              </Col>
            </Row>
          </div>
        )}
      </Card>
      
      {/* 用户统计和设置卡片 */}
      <Row gutter={16} className="profile-sections">
        <Col span={12}>
          {/* 用户统计卡片 */}
          <Card className="stats-card" title="学习统计" extra={<BookOutlined />}>
            <div className="stats-grid">
              <div className="stat-item">
                <div className="stat-number">{userStats.wordsLearned}</div>
                <div className="stat-label">累计学习词语</div>
              </div>
              <div className="stat-item">
                <div className="stat-number">{userStats.exercisesCompleted}</div>
                <div className="stat-label">完成练习</div>
              </div>
              <div className="stat-item">
                <div className="stat-number">{userStats.studyDays}</div>
                <div className="stat-label">学习天数</div>
              </div>
              <div className="stat-item">
                <div className="stat-number">{userStats.achievementCount}</div>
                <div className="stat-label">获得成就</div>
              </div>
            </div>
          </Card>
          
          {/* 成就卡片 */}
          <Card className="achievements-card" title="我的成就" extra={<TrophyOutlined />}>
            <List
              dataSource={achievements}
              renderItem={(achievement) => (
                <List.Item className="achievement-item">
                  <div className="achievement-icon">{achievement.icon}</div>
                  <div className="achievement-info">
                    <div className="achievement-name">{achievement.name}</div>
                    <div className="achievement-description">{achievement.description}</div>
                    <div className="achievement-date">获得日期：{achievement.date}</div>
                  </div>
                  <Badge 
                    color={getAchievementColor(achievement.level)} 
                    text={getAchievementLevelTag(achievement.level)}
                    className="achievement-badge"
                  />
                </List.Item>
              )}
              locale={{ emptyText: '暂无成就' }}
            />
          </Card>
        </Col>
        
        <Col span={12}>
          {/* 账户设置卡片 */}
          <Card className="settings-card" title="账户设置" extra={<SettingOutlined />}>
            <div className="settings-list">
              <div className="setting-item">
                <div className="setting-info">
                  <div className="setting-name">修改密码</div>
                  <div className="setting-description">定期修改密码可以保障账户安全</div>
                </div>
                <Button onClick={openPasswordModal} type="text">
                  修改
                </Button>
              </div>
              
              <Divider />
              
              <div className="setting-item">
                <div className="setting-info">
                  <div className="setting-name">每日学习提醒</div>
                  <div className="setting-description">开启后将每日提醒您进行学习</div>
                </div>
                <Switch 
                  checked={settings.dailyReminder} 
                  onChange={(checked) => updateSetting('dailyReminder', checked)}
                />
              </div>
              
              <div className="setting-item">
                <div className="setting-info">
                  <div className="setting-name">通知提醒</div>
                  <div className="setting-description">接收系统通知和学习提醒</div>
                </div>
                <Switch 
                  checked={settings.notificationEnabled} 
                  onChange={(checked) => updateSetting('notificationEnabled', checked)}
                />
              </div>
              
              <div className="setting-item">
                <div className="setting-info">
                  <div className="setting-name">暗黑模式</div>
                  <div className="setting-description">切换暗黑/明亮显示模式</div>
                </div>
                <Switch 
                  checked={settings.darkMode} 
                  onChange={(checked) => updateSetting('darkMode', checked)}
                />
              </div>
              
              <div className="setting-item">
                <div className="setting-info">
                  <div className="setting-name">语言设置</div>
                  <div className="setting-description">选择系统显示语言</div>
                </div>
                <Select 
                  value={settings.language} 
                  style={{ width: 120 }} 
                  onChange={(value) => updateSetting('language', value)}
                >
                  <Option value="zh-CN">简体中文</Option>
                  <Option value="en-US">English</Option>
                </Select>
              </div>
            </div>
          </Card>
          
          {/* 安全提示卡片 */}
          <Card className="security-card" title="安全提示">
            <div className="security-tips">
              <div className="tip-item">
                <Badge status="warning" className="tip-badge" />
                <span>请定期修改密码以保障账户安全</span>
              </div>
              <div className="tip-item">
                <Badge status="info" className="tip-badge" />
                <span>不要与他人分享您的账号信息</span>
              </div>
              <div className="tip-item">
                <Badge status="info" className="tip-badge" />
                <span>请确保您的邮箱和手机号是最新的</span>
              </div>
            </div>
          </Card>
        </Col>
      </Row>
      
      {/* 修改密码弹窗 */}
      <Modal
        title="修改密码"
        open={showPasswordModal}
        onCancel={closePasswordModal}
        footer={null}
      >
        <Form form={passwordForm} layout="vertical" onFinish={changePassword}>
          <Form.Item 
            label="当前密码" 
            name="currentPassword" 
            rules={[{ required: true, message: '请输入当前密码' }]}
          >
            <Input.Password placeholder="请输入当前密码" prefix={<LockOutlined />} />
          </Form.Item>
          <Form.Item 
            label="新密码" 
            name="newPassword" 
            rules={[
              { required: true, message: '请输入新密码' },
              { min: 6, message: '密码长度不能少于6位' }
            ]}
          >
            <Input.Password placeholder="请输入新密码" prefix={<LockOutlined />} />
          </Form.Item>
          <Form.Item 
            label="确认新密码" 
            name="confirmPassword" 
            rules={[
              { required: true, message: '请确认新密码' },
              ({ getFieldValue }) => ({
                validator(rule, value) {
                  if (!value || getFieldValue('newPassword') === value) {
                    return Promise.resolve();
                  }
                  return Promise.reject('两次输入的密码不一致');
                },
              }),
            ]}
          >
            <Input.Password placeholder="请再次输入新密码" prefix={<LockOutlined />} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" style={{ width: '100%' }}>
              确认修改
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}



export default Profile