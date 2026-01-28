// 用户服务模块

class UserService {
  constructor() {
    // 初始化服务配置
    this.config = {
      apiBaseUrl: '/api/users',
      timeout: 10000,
      cacheEnabled: true,
      cacheDuration: 300000 // 5分钟缓存时间
    }
    
    // 缓存管理
    this.cache = new Map()
    
    // 初始化服务
    this.initialize()
  }
  
  // 初始化服务
  initialize() {
    // 可以在这里设置一些初始化配置
    // 例如，从本地存储中读取用户配置
    const savedConfig = localStorage.getItem('userServiceConfig')
    if (savedConfig) {
      try {
        this.config = { ...this.config, ...JSON.parse(savedConfig) }
      } catch (error) {
        console.error('Failed to parse saved user service config:', error)
      }
    }
  }
  
  // 保存服务配置到本地存储
  saveConfig() {
    try {
      localStorage.setItem('userServiceConfig', JSON.stringify(this.config))
    } catch (error) {
      console.error('Failed to save user service config:', error)
    }
  }
  
  // 获取缓存数据
  getCachedData(key) {
    if (!this.config.cacheEnabled) {
      return null
    }
    
    const cached = this.cache.get(key)
    if (!cached) {
      return null
    }
    
    // 检查缓存是否过期
    if (Date.now() > cached.expiry) {
      this.cache.delete(key)
      return null
    }
    
    return cached.data
  }
  
  // 设置缓存数据
  setCachedData(key, data) {
    if (!this.config.cacheEnabled) {
      return
    }
    
    this.cache.set(key, {
      data,
      expiry: Date.now() + this.config.cacheDuration
    })
  }
  
  // 清除缓存
  clearCache() {
    this.cache.clear()
  }
  
  // 获取用户信息
  async getUserInfo(userId = 'current') {
    const cacheKey = `userInfo_${userId}`
    const cachedData = this.getCachedData(cacheKey)
    
    if (cachedData) {
      return cachedData
    }
    
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await axios.get(`${this.config.apiBaseUrl}/${userId}`)
      // const userInfo = response.data
      
      // 模拟API调用和数据返回
      const userInfo = await this.mockGetUserInfo(userId)
      
      // 缓存数据
      this.setCachedData(cacheKey, userInfo)
      
      return userInfo
    } catch (error) {
      console.error('Failed to get user info:', error)
      throw error
    }
  }
  
  // 更新用户信息
  async updateUserInfo(userId = 'current', userData) {
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await axios.put(`${this.config.apiBaseUrl}/${userId}`, userData)
      // const updatedUserInfo = response.data
      
      // 模拟API调用和数据返回
      const updatedUserInfo = await this.mockUpdateUserInfo(userId, userData)
      
      // 清除缓存，下次获取时重新加载
      this.cache.delete(`userInfo_${userId}`)
      
      return updatedUserInfo
    } catch (error) {
      console.error('Failed to update user info:', error)
      throw error
    }
  }
  
  // 修改密码
  async changePassword(oldPassword, newPassword) {
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await axios.post(`${this.config.apiBaseUrl}/change-password`, {
      //   oldPassword,
      //   newPassword
      // })
      
      // 模拟API调用
      await this.mockChangePassword(oldPassword, newPassword)
      
      return { success: true, message: '密码修改成功' }
    } catch (error) {
      console.error('Failed to change password:', error)
      throw error
    }
  }
  
  // 上传头像
  async uploadAvatar(avatarFile) {
    try {
      // 在实际环境中，这里会调用真实的API上传文件
      // const formData = new FormData()
      // formData.append('avatar', avatarFile)
      // const response = await axios.post(`${this.config.apiBaseUrl}/upload-avatar`, formData)
      
      // 模拟API调用和返回新的头像URL
      const newAvatarUrl = await this.mockUploadAvatar(avatarFile)
      
      // 清除用户信息缓存，下次获取时重新加载
      this.cache.delete('userInfo_current')
      
      return newAvatarUrl
    } catch (error) {
      console.error('Failed to upload avatar:', error)
      throw error
    }
  }
  
  // 获取用户统计数据
  async getUserStats(userId = 'current') {
    const cacheKey = `userStats_${userId}`
    const cachedData = this.getCachedData(cacheKey)
    
    if (cachedData) {
      return cachedData
    }
    
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await axios.get(`${this.config.apiBaseUrl}/${userId}/stats`)
      // const stats = response.data
      
      // 模拟API调用和数据返回
      const stats = await this.mockGetUserStats(userId)
      
      // 缓存数据
      this.setCachedData(cacheKey, stats)
      
      return stats
    } catch (error) {
      console.error('Failed to get user stats:', error)
      throw error
    }
  }
  
  // 获取用户成就
  async getUserAchievements(userId = 'current') {
    const cacheKey = `userAchievements_${userId}`
    const cachedData = this.getCachedData(cacheKey)
    
    if (cachedData) {
      return cachedData
    }
    
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await axios.get(`${this.config.apiBaseUrl}/${userId}/achievements`)
      // const achievements = response.data
      
      // 模拟API调用和数据返回
      const achievements = await this.mockGetUserAchievements(userId)
      
      // 缓存数据
      this.setCachedData(cacheKey, achievements)
      
      return achievements
    } catch (error) {
      console.error('Failed to get user achievements:', error)
      throw error
    }
  }
  
  // 获取用户设置
  async getUserSettings(userId = 'current') {
    const cacheKey = `userSettings_${userId}`
    const cachedData = this.getCachedData(cacheKey)
    
    if (cachedData) {
      return cachedData
    }
    
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await axios.get(`${this.config.apiBaseUrl}/${userId}/settings`)
      // const settings = response.data
      
      // 模拟API调用和数据返回
      const settings = await this.mockGetUserSettings(userId)
      
      // 缓存数据
      this.setCachedData(cacheKey, settings)
      
      return settings
    } catch (error) {
      console.error('Failed to get user settings:', error)
      throw error
    }
  }
  
  // 更新用户设置
  async updateUserSettings(userId = 'current', settingsData) {
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await axios.put(`${this.config.apiBaseUrl}/${userId}/settings`, settingsData)
      // const updatedSettings = response.data
      
      // 模拟API调用和数据返回
      const updatedSettings = await this.mockUpdateUserSettings(userId, settingsData)
      
      // 清除缓存
      this.cache.delete(`userSettings_${userId}`)
      
      return updatedSettings
    } catch (error) {
      console.error('Failed to update user settings:', error)
      throw error
    }
  }
  
  // 注册新用户
  async registerUser(userData) {
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await axios.post(`${this.config.apiBaseUrl}/register`, userData)
      // const newUser = response.data
      
      // 模拟API调用和数据返回
      const newUser = await this.mockRegisterUser(userData)
      
      return newUser
    } catch (error) {
      console.error('Failed to register user:', error)
      throw error
    }
  }
  
  // 用户登录
  async login(credentials) {
    try {
      // 在实际环境中，这里会调用真实的API
      // const response = await axios.post(`${this.config.apiBaseUrl}/login`, credentials)
      // const loginResult = response.data
      
      // 模拟API调用和数据返回
      const loginResult = await this.mockLogin(credentials)
      
      // 如果登录成功，保存token等信息
      if (loginResult.success && loginResult.token) {
        localStorage.setItem('authToken', loginResult.token)
        localStorage.setItem('currentUser', JSON.stringify(loginResult.user))
        
        // 清除缓存，以便重新加载用户数据
        this.clearCache()
      }
      
      return loginResult
    } catch (error) {
      console.error('Login failed:', error)
      throw error
    }
  }
  
  // 用户登出
  async logout() {
    try {
      // 在实际环境中，这里会调用真实的API
      // await axios.post(`${this.config.apiBaseUrl}/logout`)
      
      // 模拟API调用
      await this.mockLogout()
      
      // 清除本地存储中的用户信息和token
      localStorage.removeItem('authToken')
      localStorage.removeItem('currentUser')
      
      // 清除缓存
      this.clearCache()
      
      return { success: true }
    } catch (error) {
      console.error('Logout failed:', error)
      throw error
    }
  }
  
  // 检查用户是否已登录
  isLoggedIn() {
    const token = localStorage.getItem('authToken')
    return !!token
  }
  
  // 获取当前登录用户
  getCurrentUser() {
    const userStr = localStorage.getItem('currentUser')
    if (userStr) {
      try {
        return JSON.parse(userStr)
      } catch (error) {
        console.error('Failed to parse current user:', error)
        return null
      }
    }
    return null
  }
  
  // 获取认证token
  getAuthToken() {
    return localStorage.getItem('authToken')
  }
  
  // Mock函数 - 获取用户信息
  async mockGetUserInfo(userId) {
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 400))
    
    // 返回模拟数据
    return {
      id: userId === 'current' ? '1001' : userId,
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
  }
  
  // Mock函数 - 更新用户信息
  async mockUpdateUserInfo(userId, userData) {
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 500))
    
    // 合并原有数据和新数据
    const originalData = await this.mockGetUserInfo(userId)
    const updatedData = { ...originalData, ...userData }
    
    return updatedData
  }
  
  // Mock函数 - 修改密码
  async mockChangePassword(oldPassword, newPassword) {
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 500))
    
    // 简单的密码验证逻辑（实际应用中应该更复杂）
    if (!oldPassword || !newPassword || newPassword.length < 6) {
      throw new Error('密码不符合要求')
    }
    
    // 模拟修改成功
    return { success: true }
  }
  
  // Mock函数 - 上传头像
  async mockUploadAvatar(avatarFile) {
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // 生成随机的头像URL作为模拟上传后的结果
    return `https://api.dicebear.com/6.x/avataaars/svg?seed=${Math.random()}`
  }
  
  // Mock函数 - 获取用户统计数据
  async mockGetUserStats(userId) {
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 300))
    
    // 返回模拟数据
    return {
      wordsLearned: 320,
      exercisesCompleted: 15,
      studyDays: 45,
      achievementCount: 8,
      averageScore: 85.5,
      recentStudyTime: 3600 // 秒
    }
  }
  
  // Mock函数 - 获取用户成就
  async mockGetUserAchievements(userId) {
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 350))
    
    // 返回模拟数据
    return [
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
  }
  
  // Mock函数 - 获取用户设置
  async mockGetUserSettings(userId) {
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 250))
    
    // 返回模拟数据
    return {
      dailyReminder: true,
      notificationEnabled: true,
      darkMode: false,
      language: 'zh-CN',
      soundEffects: true,
      autoPlayVoice: true,
      learningGoal: 30 // 每天学习目标30分钟
    }
  }
  
  // Mock函数 - 更新用户设置
  async mockUpdateUserSettings(userId, settingsData) {
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 300))
    
    // 合并原有设置和新设置
    const originalSettings = await this.mockGetUserSettings(userId)
    const updatedSettings = { ...originalSettings, ...settingsData }
    
    return updatedSettings
  }
  
  // Mock函数 - 注册用户
  async mockRegisterUser(userData) {
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 600))
    
    // 简单的验证逻辑
    if (!userData.name || !userData.email || !userData.password) {
      throw new Error('注册信息不完整')
    }
    
    // 返回模拟的新用户数据
    return {
      id: `user_${Date.now()}`,
      name: userData.name,
      email: userData.email,
      createdAt: new Date().toISOString()
    }
  }
  
  // Mock函数 - 用户登录
  async mockLogin(credentials) {
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 500))
    
    // 简单的登录验证逻辑
    if (credentials.username === 'admin' && credentials.password === 'password') {
      // 模拟登录成功
      const user = {
        id: '1001',
        name: '张三',
        role: 'admin'
      }
      
      return {
        success: true,
        token: `mock_token_${Date.now()}`,
        user
      }
    } else {
      // 模拟登录失败
      throw new Error('用户名或密码错误')
    }
  }
  
  // Mock函数 - 用户登出
  async mockLogout() {
    // 模拟网络延迟
    await new Promise(resolve => setTimeout(resolve, 200))
    
    // 模拟登出成功
    return { success: true }
  }
}

// 创建并导出userService实例
export const userService = new UserService()

export default userService