# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码仓库中工作时提供指导。

## 项目概述

这是一个 AI 语文助手 - 一个专为体育生设计的语文学习辅助 Web 应用，旨在帮助体育生更高效地提升语文基础水平。这是一个基于 React 和 Vite 构建的单页应用 (SPA)。

## 常用开发命令

### 开发环境
```bash
npm run dev      # 启动开发服务器，端口 3000，自动打开浏览器
```

### 构建和部署
```bash
npm run build    # 生产环境构建（输出到 dist/ 文件夹）
npm run preview  # 本地预览生产构建
```

### 依赖管理
```bash
npm install      # 安装所有依赖
```

## 架构与技术栈

### 核心技术
- **React 18.2.0** - UI 框架，使用函数组件和 Hooks
- **Vite 4.4.0** - 构建工具和开发服务器
- **React Router DOM 6.14.2** - 客户端路由
- **Ant Design 5.7.0** - UI 组件库，已配置中文本地化 (zhCN)
- **@ant-design/plots 2.6.3** - 数据可视化图表库

### API 集成
- **Axios 1.4.0** - HTTP 客户端
- **DeepSeek API** - AI 聊天功能，用于中文语言辅导

## 项目结构

```
src/
├── pages/              # 13个页面组件
│   ├── Home.jsx        # 首页/仪表板
│   ├── Vocabulary.jsx  # 词语学习
│   ├── Literature.jsx  # 古诗词学习
│   ├── Idiom.jsx       # 熟语习语学习
│   ├── Exercise.jsx    # 练习中心
│   ├── ExerciseDetail.jsx # 练习详情
│   ├── Pinyin.jsx      # 拼音学习
│   ├── Correction.jsx  # 文本纠错
│   ├── Chat.jsx        # AI 聊天界面
│   ├── StudyRecord.jsx # 学习记录
│   ├── StudyReport.jsx # 学习报告（含图表）
│   ├── StudyAnalysis.jsx # 学习分析
│   └── Profile.jsx     # 个人资料
│
├── services/           # API 服务层
│   ├── aiChatService.js     # DeepSeek AI 集成（含缓存）
│   ├── vocabularyService.js # 词汇数据管理
│   ├── studyRecordService.js # 学习记录管理
│   └── userService.js       # 用户管理
│
├── App.jsx            # 主应用，包含路由和导航
├── App.css            # 应用样式
├── main.jsx           # 入口文件
└── index.css          # 全局样式
```

## 核心架构模式

### 1. 服务层模式 (Service Layer Pattern)
- 所有 API 调用封装在 `src/services/` 下的服务类中
- 每个服务处理特定领域（AI 聊天、词汇、学习记录、用户管理）
- 服务使用单例模式（导出实例而非类）

### 2. 组件化架构
- React 函数组件和 Hooks
- 页面和可复用组件清晰分离
- 全站统一使用 Ant Design 组件

### 3. 路由结构
- 使用 browser history 的客户端路由
- 导航菜单带激活状态跟踪
- 在 App.jsx 中进行路由映射

### 4. AI 集成
- 通过 `aiChatService.js` 集成 DeepSeek API
- 可配置参数（模型、温度、最大令牌数）
- 内置 TTL 缓存机制
- 错误处理和超时管理

## 环境配置

在 `.env` 文件中配置必要的环境变量：
```
VITE_DEEPSEEK_API_KEY=你的API密钥
```

## 重要说明

### 无测试框架
- 目前未配置单元测试、集成测试或 E2E 测试
- 依赖中没有测试库

### 导航实现
- 使用 `window.location.href` 进行导航（而非 React Router 的 Navigate 组件）
- 这会导致完整的页面刷新，而非 SPA 风格的导航

### UI 本地化
- 所有 UI 文本和内容均为中文
- Ant Design 已配置中文语言环境 (`zhCN`)

### API 配置
- DeepSeek API 基础 URL: `https://api.deepseek.com/v1`
- 默认模型: `deepseek-chat`
- 可配置超时: 10 秒
- 缓存 TTL: 1 小时

## 开发指南

### 添加新页面
1. 在 `src/pages/` 中创建页面组件
2. 在 App.jsx 的 Routes 组件中添加路由
3. 在头部 Menu 中添加菜单项，包含图标和 pathMap 条目
4. 在 handleMenuClick 函数中更新路径映射

### 添加新服务
1. 在 `src/services/` 中创建服务类
2. 使用单例模式（导出实例而非类）
3. 包含适当的错误处理和日志记录
4. 遵循现有命名约定 (`xxxService.js`)

### 使用 AI 功能
- 所有 AI 相关功能使用 `aiChatService`
- 根据不同用例自定义系统提示词
- 利用内置缓存处理重复请求
- 优雅地处理 API 错误，提供用户友好的消息

### 使用图表
- 使用 `@ant-design/plots` 进行数据可视化
- 导入所需的图表类型（折线图、柱状图、饼图等）
- 遵循 StudyReport 和 StudyAnalysis 页面中的现有图表模式