import React, { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { Layout, Menu, message } from 'antd'
import {
  HomeOutlined,
  BookOutlined,
  LineChartOutlined,
  BarChartOutlined,
  FileTextOutlined,
  UserOutlined,
  LogoutOutlined,
  LoginOutlined
} from '@ant-design/icons'
import { authService } from './services/authService'
import Home from './pages/Home'
import Exercise from './pages/Exercise'
import StudyReport from './pages/StudyReport'
import StudyAnalysis from './pages/StudyAnalysis'
import StudyRecord from './pages/StudyRecord'
import Profile from './pages/Profile'
import Vocabulary from './pages/Vocabulary'
import Literature from './pages/Literature'
import Pinyin from './pages/Pinyin'
import Correction from './pages/Correction'
import Chat from './pages/Chat'
import Idiom from './pages/Idiom'
import ExerciseDetail from './pages/ExerciseDetail'
import MockExam from './pages/MockExam'
import Login from './pages/Login'
import './App.css'

const { Header, Content, Footer } = Layout

// 路由守卫组件
function PrivateRoute({ children }) {
  const isAuthenticated = authService.isLoggedIn();

  return isAuthenticated ? children : <Navigate to="/login" replace />;
}

function App() {
  const [current, setCurrent] = useState('home')
  const [isLogin, setIsLogin] = useState(false)
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  // 检查登录状态
  useEffect(() => {
    const checkLoginStatus = async () => {
      try {
        const loggedIn = authService.isLoggedIn();
        if (loggedIn) {
          const currentUser = authService.getStoredUser();
          if (currentUser) {
            setIsLogin(true);
            setUser(currentUser);
          } else {
            // 如果有token但没有用户信息，尝试获取
            const result = await authService.getCurrentUser();
            if (result.success) {
              setIsLogin(true);
              setUser(result.user);
            }
          }
        }
      } catch (error) {
        console.error('检查登录状态失败:', error);
      } finally {
        setLoading(false);
      }
    };

    checkLoginStatus();
  }, []);

  const handleMenuClick = (e) => {
    if (e.key === 'logout') {
      handleLogout();
      return;
    }

    setCurrent(e.key);
    const pathMap = {
      'home': '/home',
      'vocabulary': '/vocabulary',
      'literature': '/literature',
      'idiom': '/idiom',
      'exercise': '/exercise',
      'study-record': '/study-record',
      'study-report': '/study-report',
      'study-analysis': '/study-analysis',
      'profile': '/profile',
      'chat': '/chat'
    };

    if (pathMap[e.key]) {
      window.location.href = pathMap[e.key];
    }
  };

  const handleLogout = async () => {
    try {
      const result = await authService.logout();
      if (result.success) {
        message.success('退出登录成功');
        setIsLogin(false);
        setUser(null);
        window.location.href = '/login';
      }
    } catch (error) {
      message.error('退出登录失败');
      console.error('Logout error:', error);
    }
  };

  // 显示加载状态
  if (loading) {
    return (
      <div style={{
        height: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '16px'
      }}>
        加载中...
      </div>
    );
  }

  return (
    <Router>
      <Layout className="layout">
        <Header className="header">
          <div className="logo">AI语文助手</div>
          {isLogin && (
            <Menu
              theme="dark"
              mode="horizontal"
              selectedKeys={[current]}
              onClick={handleMenuClick}
              className="header-menu"
            >
              <Menu.Item key="home" icon={<HomeOutlined />}>
                首页
              </Menu.Item>
              <Menu.Item key="vocabulary" icon={<BookOutlined />}>
                词语学习
              </Menu.Item>
              <Menu.Item key="literature" icon={<BookOutlined />}>
                古诗词学习
              </Menu.Item>
              <Menu.Item key="idiom" icon={<BookOutlined />}>
                熟语习语
              </Menu.Item>
              <Menu.Item key="exercise" icon={<FileTextOutlined />}>
                练习中心
              </Menu.Item>
              <Menu.Item key="chat" icon={<FileTextOutlined />}>
                AI聊天
              </Menu.Item>
              <Menu.Item key="study-record" icon={<LineChartOutlined />}>
                学习记录
              </Menu.Item>
              <Menu.Item key="study-report" icon={<BarChartOutlined />}>
                学习报告
              </Menu.Item>
              <Menu.Item key="study-analysis" icon={<BarChartOutlined />}>
                学习分析
              </Menu.Item>
              <Menu.Item key="profile" icon={<UserOutlined />}>
                个人资料
              </Menu.Item>
              <Menu.Item key="logout" icon={<LogoutOutlined />}>
                退出登录
              </Menu.Item>
            </Menu>
          )}
        </Header>
        
        <Content className="content">
          <div className="container">
            <Routes>
              {/* 登录页面路由 */}
              <Route path="/login" element={<Login />} />

              {/* 需要登录才能访问的路由 */}
              <Route path="/" element={
                <PrivateRoute>
                  <Home />
                </PrivateRoute>
              } />
              <Route path="/home" element={
                <PrivateRoute>
                  <Home />
                </PrivateRoute>
              } />
              <Route path="/exercise" element={
                <PrivateRoute>
                  <Exercise />
                </PrivateRoute>
              } />
              <Route path="/exercise-detail/:id" element={
                <PrivateRoute>
                  <ExerciseDetail />
                </PrivateRoute>
              } />
              <Route path="/study-report" element={
                <PrivateRoute>
                  <StudyReport />
                </PrivateRoute>
              } />
              <Route path="/report" element={
                <PrivateRoute>
                  <StudyReport />
                </PrivateRoute>
              } />
              <Route path="/study-analysis" element={
                <PrivateRoute>
                  <StudyAnalysis />
                </PrivateRoute>
              } />
              <Route path="/study-record" element={
                <PrivateRoute>
                  <StudyRecord />
                </PrivateRoute>
              } />
              <Route path="/profile" element={
                <PrivateRoute>
                  <Profile />
                </PrivateRoute>
              } />
              <Route path="/vocabulary" element={
                <PrivateRoute>
                  <Vocabulary />
                </PrivateRoute>
              } />
              <Route path="/literature" element={
                <PrivateRoute>
                  <Literature />
                </PrivateRoute>
              } />
              <Route path="/poetry" element={
                <PrivateRoute>
                  <Literature />
                </PrivateRoute>
              } />
              <Route path="/pinyin" element={
                <PrivateRoute>
                  <Pinyin />
                </PrivateRoute>
              } />
              <Route path="/correction" element={
                <PrivateRoute>
                  <Correction />
                </PrivateRoute>
              } />
              <Route path="/chat" element={
                <PrivateRoute>
                  <Chat />
                </PrivateRoute>
              } />
              <Route path="/mock-exam" element={
                <PrivateRoute>
                  <MockExam />
                </PrivateRoute>
              } />
              <Route path="/idiom" element={
                <PrivateRoute>
                  <Idiom />
                </PrivateRoute>
              } />

              {/* 默认重定向到首页 */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </Content>

        <Footer className="footer">
          <div className="footer-content">
            <div className="contact-info">
              <p>联系我们：insistgang@163.com</p>
              <p>版本：v3.0.0</p>
            </div>
            <div className="copyright">
              © 2025 AI语文助手 - 体育生语文基础智能辅导系统
            </div>
          </div>
        </Footer>
      </Layout>
    </Router>
  )
}

export default App