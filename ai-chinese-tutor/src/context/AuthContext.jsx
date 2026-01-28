import React, { createContext, useContext, useState, useEffect } from 'react';
import { authService } from '../services/authService';
import { message } from 'antd';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkLoginStatus = async () => {
      try {
        const loggedIn = authService.isLoggedIn();
        if (loggedIn) {
          const currentUser = authService.getStoredUser();
          if (currentUser) {
            setUser(currentUser);
          } else {
            // 如果有token但没有用户信息，尝试获取
            const result = await authService.getCurrentUser();
            if (result.success) {
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

  const login = async (username, password) => {
    try {
      const result = await authService.login(username, password);

      if (result.success) {
        setUser(result.user);
        message.success('登录成功！');
        return { success: true };
      } else {
        message.error(result.message || '登录失败');
        return { success: false, message: result.message };
      }
    } catch (error) {
      message.error('登录失败，请稍后重试');
      console.error('Login error:', error);
      return { success: false, message: '登录失败，请稍后重试' };
    }
  };

  const register = async (userData) => {
    try {
      const result = await authService.register(userData);

      if (result.success) {
        setUser(result.user);
        message.success('注册成功！');
        return { success: true };
      } else {
        message.error(result.message || '注册失败');
        return { success: false, message: result.message };
      }
    } catch (error) {
      message.error('注册失败，请稍后重试');
      console.error('Register error:', error);
      return { success: false, message: '注册失败，请稍后重试' };
    }
  };

  const logout = async () => {
    try {
      const result = await authService.logout();
      if (result.success) {
        setUser(null);
        message.success('退出登录成功');
        return { success: true };
      }
    } catch (error) {
      message.error('退出登录失败');
      console.error('Logout error:', error);
      // 即使接口调用失败，也清除本地状态
      setUser(null);
      return { success: true };
    }
  };

  const updateProfile = async (userData) => {
    try {
      const result = await authService.updateProfile(userData);

      if (result.success) {
        setUser(result.user);
        message.success('更新成功！');
        return { success: true, user: result.user };
      } else {
        message.error(result.message || '更新失败');
        return { success: false, message: result.message };
      }
    } catch (error) {
      message.error('更新失败，请稍后重试');
      console.error('Update profile error:', error);
      return { success: false, message: '更新失败，请稍后重试' };
    }
  };

  const value = {
    user,
    login,
    register,
    logout,
    updateProfile,
    loading,
    isLoggedIn: !!user,
    isTokenValid: authService.isTokenValid()
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default AuthContext;