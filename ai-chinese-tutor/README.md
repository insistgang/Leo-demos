# AI语文助手 - 体育生语文基础智能辅导系统

## 项目简介
AI语文助手是一款专为体育生设计的语文学习辅助Web应用，旨在帮助体育生更高效地提升语文基础水平。通过个性化的学习方案、智能的词语学习、成语学习、语法练习和学习数据分析，为体育生提供全方位的语文学习支持。

## 功能特点

### 1. 词语学习
- 词语查询与学习
- 近义词辨析
- 词语练习与测试
- 词语收藏与复习

### 2. 成语学习
- 成语查询与解释
- 成语故事学习
- 成语接龙游戏
- 成语练习与测试

### 3. 语法练习
- 句子成分分析
- 病句修改练习
- 语法知识点学习

### 4. 阅读理解
- 文章阅读练习
- 阅读理解测试
- 阅读技巧指导

### 5. 学习记录与分析
- 详细学习记录
- 学习数据统计
- 学习进度追踪
- 个性化学习建议

### 6. 用户管理
- 个人资料管理
- 学习偏好设置
- 成就系统

## 技术栈

### 前端技术
- React - 构建用户界面的JavaScript库
- Ant Design - UI组件库
- React Router - 路由管理
- Axios - HTTP请求客户端
- @ant-design/plots - 数据可视化图表库
- Vite - 构建工具

## 快速开始

### 前提条件
确保您的系统已安装以下软件：
- Node.js (14.x 或更高版本)
- npm 或 yarn 包管理器

### 安装步骤

1. 克隆项目代码
```bash
# 克隆项目
git clone <项目仓库地址>

# 进入项目目录
cd ai-chinese-tutor-web
```

2. 安装依赖
```bash
# 使用npm
npm install

# 或使用yarn
yarn install
```

3. 启动开发服务器
```bash
# 使用npm
npm run dev

# 或使用yarn
yarn dev
```

4. 构建生产版本
```bash
# 使用npm
npm run build

# 或使用yarn
yarn build
```

5. 预览生产版本
```bash
# 使用npm
npm run preview

# 或使用yarn
yarn preview
```

## 项目结构

```
src/
  ├── pages/             # 页面组件
  │   ├── Home.jsx       # 首页
  │   ├── Vocabulary.jsx # 词语学习页
  │   ├── Exercise.jsx   # 练习页面
  │   ├── StudyRecord.jsx # 学习记录页
  │   ├── StudyReport.jsx # 学习报告页
  │   ├── StudyAnalysis.jsx # 学习分析页
  │   └── Profile.jsx    # 个人资料页
  ├── services/          # API服务
  │   ├── vocabularyService.js # 词语学习服务
  │   ├── studyRecordService.js # 学习记录服务
  │   └── userService.js # 用户服务
  ├── components/        # 通用组件
  ├── utils/             # 工具函数
  ├── assets/            # 静态资源
  ├── App.jsx            # 应用入口组件
  ├── App.css            # 应用样式
  ├── main.jsx           # 应用主文件
  └── index.css          # 全局样式
```

## 开发说明

### 组件开发
- 所有页面组件位于 `src/pages/` 目录
- 可复用组件位于 `src/components/` 目录
- 使用Ant Design组件库构建UI界面

### API服务
- 所有API调用封装在 `src/services/` 目录下的服务类中
- 目前使用模拟数据，可根据实际后端API进行替换

### 样式规范
- 使用CSS Modules或styled-components管理组件样式
- 全局样式定义在 `src/index.css`
- 应用级样式定义在 `src/App.css`

## 浏览器支持
- Chrome (最新版本)
- Firefox (最新版本)
- Safari (最新版本)
- Edge (最新版本)

## 许可证
本项目采用MIT许可证 - 详情请查看LICENSE文件

## 联系方式
如有问题或建议，请联系项目维护人员。

## 更新日志

### 版本 1.0.0
- 初始版本发布
- 实现词语学习、成语学习、语法练习等核心功能
- 实现学习记录、学习报告和学习分析功能
- 实现用户管理和个人资料功能

## 特别鸣谢
感谢所有为项目做出贡献的开发人员和用户。