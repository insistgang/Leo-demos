import api from './api';

class AuthService {
  // 用户登录
  async login(username, password) {
    try {
      const response = await api.post('/auth/login', {
        username,
        password
      });

      if (response.success) {
        // 保存 token 和用户信息
        localStorage.setItem('authToken', response.token);
        localStorage.setItem('currentUser', JSON.stringify(response.user));

        return {
          success: true,
          user: response.user
        };
      }

      return {
        success: false,
        message: response.message || '登录失败'
      };
    } catch (error) {
      return {
        success: false,
        message: error.response?.data?.message || error.message || '登录失败，请稍后重试'
      };
    }
  }

  // 用户注册
  async register(userData) {
    try {
      const response = await api.post('/auth/register', userData);

      if (response.success) {
        // 保存 token 和用户信息
        localStorage.setItem('authToken', response.token);
        localStorage.setItem('currentUser', JSON.stringify(response.user));

        return {
          success: true,
          user: response.user
        };
      }

      return {
        success: false,
        message: response.message || '注册失败'
      };
    } catch (error) {
      return {
        success: false,
        message: error.response?.data?.message || error.message || '注册失败，请稍后重试'
      };
    }
  }

  // 用户登出
  async logout() {
    try {
      // 调用后端登出接口（可选）
      // await api.post('/auth/logout');

      // 清除本地存储
      localStorage.removeItem('authToken');
      localStorage.removeItem('currentUser');

      return {
        success: true,
        message: '登出成功'
      };
    } catch (error) {
      // 即使接口调用失败，也清除本地存储
      localStorage.removeItem('authToken');
      localStorage.removeItem('currentUser');

      return {
        success: true,
        message: '登出成功'
      };
    }
  }

  // 获取当前用户信息
  async getCurrentUser() {
    try {
      const response = await api.get('/auth/profile');

      if (response.success) {
        // 更新本地存储的用户信息
        localStorage.setItem('currentUser', JSON.stringify(response.user));

        return {
          success: true,
          user: response.user
        };
      }

      return {
        success: false,
        message: response.message || '获取用户信息失败'
      };
    } catch (error) {
      return {
        success: false,
        message: error.response?.data?.message || error.message || '获取用户信息失败'
      };
    }
  }

  // 更新用户信息
  async updateProfile(userData) {
    try {
      const response = await api.put('/auth/profile', userData);

      if (response.success) {
        // 更新本地存储的用户信息
        localStorage.setItem('currentUser', JSON.stringify(response.user));

        return {
          success: true,
          user: response.user,
          message: response.message || '更新成功'
        };
      }

      return {
        success: false,
        message: response.message || '更新失败'
      };
    } catch (error) {
      return {
        success: false,
        message: error.response?.data?.message || error.message || '更新失败，请稍后重试'
      };
    }
  }

  // 修改密码
  async changePassword(oldPassword, newPassword) {
    try {
      const response = await api.put('/auth/password', {
        oldPassword,
        newPassword
      });

      return {
        success: true,
        message: response.message || '密码修改成功'
      };
    } catch (error) {
      return {
        success: false,
        message: error.response?.data?.message || error.message || '密码修改失败，请稍后重试'
      };
    }
  }

  // 检查用户是否已登录
  isLoggedIn() {
    const token = localStorage.getItem('authToken');
    const user = localStorage.getItem('currentUser');
    return !!(token && user);
  }

  // 获取当前登录用户（从本地存储）
  getStoredUser() {
    const userStr = localStorage.getItem('currentUser');
    if (userStr) {
      try {
        return JSON.parse(userStr);
      } catch (error) {
        console.error('解析用户信息失败:', error);
        return null;
      }
    }
    return null;
  }

  // 获取认证 Token
  getToken() {
    return localStorage.getItem('authToken');
  }

  // 验证 Token 是否有效（简单验证，实际有效性由后端判断）
  isTokenValid() {
    const token = this.getToken();
    if (!token) return false;

    try {
      // 解析 JWT Token 的 payload 部分
      const payload = JSON.parse(atob(token.split('.')[1]));
      const currentTime = Date.now() / 1000;

      // 检查是否过期
      return payload.exp > currentTime;
    } catch (error) {
      console.error('Token 验证失败:', error);
      return false;
    }
  }

  // 刷新用户信息
  async refreshUserInfo() {
    if (this.isLoggedIn()) {
      return await this.getCurrentUser();
    }
    return { success: false, message: '用户未登录' };
  }
}

// 创建并导出 authService 实例
export const authService = new AuthService();

export default authService;