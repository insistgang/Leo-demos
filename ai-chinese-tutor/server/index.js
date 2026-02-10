require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const { sequelize } = require('./models');

// 导入路由
const authRoutes = require('./routes/auth');
const chatRoutes = require('./routes/chat');
const studyRoutes = require('./routes/study');
const questionRoutes = require('./routes/questions');

const app = express();
const PORT = process.env.PORT || 3001;

// 中间件
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// 日志中间件
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

// API 路由
app.use('/api/auth', authRoutes);
app.use('/api/chat', chatRoutes);
app.use('/api/study', studyRoutes);
app.use('/api/questions', questionRoutes);

// 健康检查接口
app.get('/api/health', (req, res) => {
  res.json({
    success: true,
    message: 'Server is running',
    timestamp: new Date().toISOString()
  });
});

// 404 处理
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: '接口不存在'
  });
});

// 全局错误处理
app.use((err, req, res, next) => {
  console.error('服务器错误:', err);
  res.status(500).json({
    success: false,
    message: '服务器内部错误'
  });
});

// 数据库同步和服务器启动
async function startServer() {
  try {
    // 同步数据库
    await sequelize.authenticate();
    console.log('数据库连接成功');

    // 创建表（如果不存在）
    await sequelize.sync({ force: false });
    console.log('数据库同步成功');

    // 启动服务器
    app.listen(PORT, '0.0.0.0', () => {
      console.log(`服务器运行在 http://localhost:${PORT}`);
      console.log(`服务器监听所有网络接口 (0.0.0.0:${PORT})`);
      console.log('API 端点：');
      console.log('  - POST /api/auth/register - 用户注册');
      console.log('  - POST /api/auth/login - 用户登录');
      console.log('  - GET  /api/auth/profile - 获取用户信息');
      console.log('  - PUT  /api/auth/profile - 更新用户信息');
      console.log('  - GET  /api/chat/history - 获取聊天历史');
      console.log('  - POST /api/chat/message - 保存聊天消息');
      console.log('  - GET  /api/study/records - 获取学习记录');
      console.log('  - POST /api/study/record - 添加学习记录');
      console.log('  - GET  /api/study/statistics - 获取学习统计');
      console.log('  - GET  /api/questions - 获取题目列表');
      console.log('  - GET  /api/questions/random - 随机获取题目');
      console.log('  - GET  /api/questions/type/:type - 按类型获取题目');
      console.log('  - GET  /api/questions/year/:year - 按年份获取题目');
    });
  } catch (error) {
    console.error('启动服务器失败:', error);
    process.exit(1);
  }
}

// 优雅关闭
process.on('SIGINT', async () => {
  console.log('\n正在关闭服务器...');
  await sequelize.close();
  console.log('数据库连接已关闭');
  process.exit(0);
});

startServer();