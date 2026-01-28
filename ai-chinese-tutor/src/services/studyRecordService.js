// 学习记录服务 - 提供学习记录管理、统计等功能
import api from './api';

class StudyRecordService {
  constructor() {
    // 服务配置
    this.config = {
      timeout: 10000,
      cacheEnabled: true,
      cacheTTL: 300000, // 5分钟缓存
    };

    // 初始化缓存
    this.cache = new Map();
  }

  // 初始化服务
  init() {
    console.log('StudyRecordService initialized');
  }

  // 获取学习记录
  async getStudyRecords(filters = {}) {
    try {
      // 构建查询参数
      const params = {
        limit: filters.pageSize || 20,
        offset: filters.page ? (filters.page - 1) * (filters.pageSize || 20) : 0,
        type: filters.type === 'all' ? undefined : filters.type,
        startDate: filters.startDate,
        endDate: filters.endDate
      };

      const response = await api.get('/study/records', { params });

      if (response.success) {
        // 转换数据格式以兼容前端
        const records = response.data.map(record => ({
          id: record.id,
          type: record.type,
          title: this.getRecordTitle(record),
          studyDate: new Date(record.createdAt).toLocaleDateString('zh-CN'),
          duration: record.duration,
          durationText: this.formatDuration(record.duration),
          score: record.score,
          totalQuestions: record.totalQuestions,
          correctAnswers: record.correctAnswers,
          completionRate: record.totalQuestions ? Math.round((record.correctAnswers / record.totalQuestions) * 100) : 100,
          content: record.content
        }));

        return {
          data: records,
          total: records.length,
          page: filters.page || 1,
          pageSize: filters.pageSize || 20
        };
      }

      return {
        data: [],
        total: 0,
        page: 1,
        pageSize: 20
      };
    } catch (error) {
      console.error('获取学习记录失败:', error);
      return {
        data: [],
        total: 0,
        page: 1,
        pageSize: 20
      };
    }
  }

  // 添加学习记录
  async addStudyRecord(recordData) {
    try {
      const response = await api.post('/study/record', recordData);

      if (response.success) {
        // 清除相关缓存
        this.clearCacheByPrefix('records_');
        this.clearCacheByPrefix('statistics_');

        return {
          success: true,
          data: response.data
        };
      }

      return {
        success: false,
        message: response.message || '添加失败'
      };
    } catch (error) {
      console.error('添加学习记录失败:', error);
      return {
        success: false,
        message: error.response?.data?.message || '添加失败，请稍后重试'
      };
    }
  }

  // 批量添加学习记录
  async addStudyRecords(records) {
    try {
      const response = await api.post('/study/records', {
        records
      });

      if (response.success) {
        // 清除相关缓存
        this.clearCacheByPrefix('records_');
        this.clearCacheByPrefix('statistics_');

        return {
          success: true,
          data: response.data
        };
      }

      return {
        success: false,
        message: response.message || '批量添加失败'
      };
    } catch (error) {
      console.error('批量添加学习记录失败:', error);
      return {
        success: false,
        message: error.response?.data?.message || '批量添加失败，请稍后重试'
      };
    }
  }

  // 删除学习记录
  async deleteStudyRecord(recordId) {
    try {
      const response = await api.delete(`/study/record/${recordId}`);

      if (response.success) {
        // 清除相关缓存
        this.clearCacheByPrefix('records_');
        this.clearCacheByPrefix('statistics_');

        return {
          success: true,
          message: response.message || '删除成功'
        };
      }

      return {
        success: false,
        message: response.message || '删除失败'
      };
    } catch (error) {
      console.error('删除学习记录失败:', error);
      return {
        success: false,
        message: error.response?.data?.message || '删除失败，请稍后重试'
      };
    }
  }

  // 获取学习统计
  async getStudyStatistics(days = 30) {
    try {
      // 构建缓存键
      const cacheKey = `statistics_${days}`;

      // 检查缓存
      if (this.config.cacheEnabled) {
        const cachedResult = this.getFromCache(cacheKey);
        if (cachedResult) {
          return cachedResult;
        }
      }

      const response = await api.get('/study/statistics', {
        params: { days }
      });

      if (response.success) {
        const statistics = response.data;

        // 转换数据格式
        const formattedStats = {
          totalStudyTime: Math.round(statistics.totalDuration / 60), // 转换为分钟
          studyDays: statistics.studyDays,
          averageDailyTime: Math.round(statistics.averageDailyDuration / 60),
          categoryData: statistics.typeStatistics.map(stat => ({
            category: this.getCategoryName(stat.type),
            studyTime: Math.round(stat.totalDuration / 60),
            count: stat.count,
            percentage: Math.round((stat.totalDuration / statistics.totalDuration) * 100)
          })),
          exerciseStats: statistics.exerciseStatistics ? {
            totalExercises: statistics.exerciseStatistics.totalExercises,
            averageScore: statistics.exerciseStatistics.averageScore,
            totalQuestions: statistics.exerciseStatistics.totalQuestions,
            totalCorrect: statistics.exerciseStatistics.totalCorrect,
            accuracy: statistics.exerciseStatistics.accuracy
          } : null
        };

        // 存入缓存
        if (this.config.cacheEnabled) {
          this.saveToCache(cacheKey, formattedStats);
        }

        return formattedStats;
      }

      // 返回默认统计数据
      return this.getDefaultStatistics();
    } catch (error) {
      console.error('获取学习统计失败:', error);
      return this.getDefaultStatistics();
    }
  }

  // 获取记录标题
  getRecordTitle(record) {
    const typeNames = {
      vocabulary: '词语学习',
      literature: '古诗词学习',
      idiom: '熟语习语',
      exercise: '练习',
      pinyin: '拼音学习',
      correction: '文本纠错'
    };

    const baseTitle = typeNames[record.type] || '学习记录';

    if (record.content && record.content.title) {
      return `${baseTitle} - ${record.content.title}`;
    }

    return baseTitle;
  }

  // 获取分类名称
  getCategoryName(type) {
    const typeNames = {
      vocabulary: '词语学习',
      literature: '古诗词',
      idiom: '熟语习语',
      exercise: '练习',
      pinyin: '拼音',
      correction: '文本纠错'
    };

    return typeNames[type] || type;
  }

  // 格式化时长
  formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;

    if (minutes < 60) {
      return `${minutes}分${remainingSeconds > 0 ? remainingSeconds + '秒' : ''}`;
    }

    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;

    return `${hours}小时${remainingMinutes > 0 ? remainingMinutes + '分钟' : ''}`;
  }

  // 获取默认统计数据
  getDefaultStatistics() {
    return {
      totalStudyTime: 0,
      studyDays: 0,
      averageDailyTime: 0,
      categoryData: [],
      exerciseStats: {
        totalExercises: 0,
        averageScore: 0,
        totalQuestions: 0,
        totalCorrect: 0,
        accuracy: 0
      }
    };
  }

  // 缓存相关方法
  getFromCache(key) {
    const cachedItem = this.cache.get(key);
    if (!cachedItem) return null;

    const now = Date.now();
    if (now > cachedItem.expiry) {
      this.cache.delete(key);
      return null;
    }

    return cachedItem.data;
  }

  saveToCache(key, data, customTTL = null) {
    const expiry = Date.now() + (customTTL || this.config.cacheTTL);
    this.cache.set(key, { data, expiry });
  }

  clearCacheByPrefix(prefix) {
    for (let key of this.cache.keys()) {
      if (key.startsWith(prefix)) {
        this.cache.delete(key);
      }
    }
  }

  clearAllCache() {
    this.cache.clear();
  }

  // 清除缓存
  clearCache() {
    this.clearAllCache();
  }

  // 获取今日得分
  async getTodayScore() {
    try {
      // 获取今天的学习记录
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      const tomorrow = new Date(today);
      tomorrow.setDate(tomorrow.getDate() + 1);

      const params = {
        startDate: today.toISOString().split('T')[0],
        endDate: today.toISOString().split('T')[0],
        limit: 1000
      };

      const response = await api.get('/study/records', { params });

      if (response.success && response.data.length > 0) {
        // 计算今日总得分
        const scores = response.data.filter(r => r.score !== null).map(r => r.score);
        if (scores.length > 0) {
          // 返回累加得分
          return scores.reduce((sum, score) => sum + score, 0);
        }
      }

      return 0;
    } catch (error) {
      console.error('获取今日得分失败:', error);
      return 0;
    }
  }

  // 获取总学习时长（分钟）
  async getTotalStudyTime() {
    try {
      const statistics = await this.getStudyStatistics(365); // 获取一年的数据
      return statistics.totalStudyTime || 0;
    } catch (error) {
      console.error('获取总学习时长失败:', error);
      return 0;
    }
  }

  // 获取已学习天数
  async getStudyDays() {
    try {
      const statistics = await this.getStudyStatistics(365);
      return statistics.studyDays || 0;
    } catch (error) {
      console.error('获取学习天数失败:', error);
      return 0;
    }
  }

  // 获取错题总数
  async getWrongCount() {
    try {
      // 获取所有练习记录
      const params = {
        limit: 10000
      };

      const response = await api.get('/study/records', { params });

      if (response.success && response.data.length > 0) {
        // 计算总错题数
        let wrongCount = 0;
        response.data.forEach(record => {
          if (record.totalQuestions && record.correctAnswers !== null) {
            wrongCount += (record.totalQuestions - record.correctAnswers);
          }
        });
        return wrongCount;
      }

      return 0;
    } catch (error) {
      console.error('获取错题数失败:', error);
      return 0;
    }
  }

  // 获取各模块学习进度
  async getProgressByType() {
    try {
      // 获取所有学习记录
      const params = { limit: 10000 };
      const response = await api.get('/study/records', { params });

      const progressList = [
        { type: 'pinyin', name: '拼音掌握度', progress: 0 },
        { type: 'vocabulary', name: '词汇量', progress: 0 },
        { type: 'literature', name: '文学常识', progress: 0 },
        { type: 'idiom', name: '成语积累', progress: 0 }
      ];

      if (response.success && response.data.length > 0) {
        // 按类型分组计算
        const typeStats = {};
        response.data.forEach(record => {
          if (!typeStats[record.type]) {
            typeStats[record.type] = {
              totalQuestions: 0,
              correctAnswers: 0,
              count: 0
            };
          }
          typeStats[record.type].totalQuestions += record.totalQuestions || 0;
          typeStats[record.type].correctAnswers += record.correctAnswers || 0;
          typeStats[record.type].count += 1;
        });

        // 计算各类型进度
        progressList.forEach(item => {
          const stats = typeStats[item.type];
          if (stats && stats.totalQuestions > 0) {
            // 基于正确率计算进度，同时考虑学习次数
            const accuracy = (stats.correctAnswers / stats.totalQuestions) * 100;
            // 学习次数越多，基础进度越高
            const baseProgress = Math.min(50, stats.count * 5);
            // 综合得分
            item.progress = Math.min(100, Math.round(baseProgress + accuracy * 0.5));
          }
        });
      }

      return progressList;
    } catch (error) {
      console.error('获取学习进度失败:', error);
      return [
        { type: 'pinyin', name: '拼音掌握度', progress: 0 },
        { type: 'vocabulary', name: '词汇量', progress: 0 },
        { type: 'literature', name: '文学常识', progress: 0 },
        { type: 'idiom', name: '成语积累', progress: 0 }
      ];
    }
  }

  // 获取首页所有统计数据
  async getHomeStatistics() {
    try {
      console.log('========== getHomeStatistics 开始 ==========');

      // 清除缓存，确保获取最新数据
      this.clearCache();

      // 一次性获取所有记录，避免多次查询
      const params = { limit: 10000 };
      console.log('请求参数:', params);
      const allRecordsResponse = await api.get('/study/records', { params });

      console.log('API响应:', allRecordsResponse);

      if (!allRecordsResponse.success) {
        console.error('获取学习记录失败');
        throw new Error('获取学习记录失败');
      }

      const allRecords = allRecordsResponse.data;
      console.log('获取到的记录数:', allRecords.length);

      // 计算今日得分
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      const tomorrow = new Date(today);
      tomorrow.setDate(tomorrow.getDate() + 1);

      const todayRecords = allRecords.filter(r => {
        const recordDate = new Date(r.createdAt);
        return recordDate >= today && recordDate < tomorrow;
      });

      const todayScore = todayRecords
        .filter(r => r.score !== null)
        .reduce((sum, r) => sum + r.score, 0);

      // 计算总学习时长（分钟）
      const totalStudyTime = Math.round(
        allRecords.reduce((sum, r) => sum + (r.duration || 0), 0) / 60
      );

      // 计算学习天数
      const uniqueDates = new Set(
        allRecords.map(r => new Date(r.createdAt).toDateString())
      );
      const studyDays = uniqueDates.size;

      // 计算错题总数
      const wrongCount = allRecords.reduce((total, record) => {
        if (record.totalQuestions && record.correctAnswers !== null) {
          return total + (record.totalQuestions - record.correctAnswers);
        }
        return total;
      }, 0);

      console.log('计算结果:');
      console.log('  今日得分:', todayScore);
      console.log('  错题数量:', wrongCount);
      console.log('  总学习时长:', totalStudyTime, '分钟');
      console.log('  学习天数:', studyDays);

      // 计算各模块进度
      const progressList = [
        { type: 'pinyin', name: '拼音掌握度', progress: 0 },
        { type: 'vocabulary', name: '词汇量', progress: 0 },
        { type: 'literature', name: '文学常识', progress: 0 },
        { type: 'idiom', name: '成语积累', progress: 0 }
      ];

      const typeStats = {};
      allRecords.forEach(record => {
        if (!typeStats[record.type]) {
          typeStats[record.type] = {
            totalQuestions: 0,
            correctAnswers: 0,
            count: 0
          };
        }
        typeStats[record.type].totalQuestions += record.totalQuestions || 0;
        typeStats[record.type].correctAnswers += record.correctAnswers || 0;
        typeStats[record.type].count += 1;
      });

      progressList.forEach(item => {
        const stats = typeStats[item.type];
        if (stats && stats.totalQuestions > 0) {
          const accuracy = (stats.correctAnswers / stats.totalQuestions) * 100;
          const baseProgress = Math.min(50, stats.count * 5);
          item.progress = Math.min(100, Math.round(baseProgress + accuracy * 0.5));
        }
      });

      const result = {
        todayScore,
        errorCount: wrongCount,
        totalStudyTime,
        studyDays,
        progressList
      };

      console.log('最终返回结果:', result);

      // 缓存结果（1小时）
      const cacheKey = 'homeStatistics';
      this.saveToCache(cacheKey, result, 60 * 60 * 1000);

      return result;
    } catch (error) {
      console.error('获取首页统计失败:', error);
      return {
        todayScore: 0,
        errorCount: 0,
        totalStudyTime: 0,
        studyDays: 0,
        progressList: [
          { type: 'pinyin', name: '拼音掌握度', progress: 0 },
          { type: 'vocabulary', name: '词汇量', progress: 0 },
          { type: 'literature', name: '文学常识', progress: 0 },
          { type: 'idiom', name: '成语积累', progress: 0 }
        ]
      };
    }
  }
}

// 创建单例实例
const studyRecordService = new StudyRecordService();

// 初始化服务
studyRecordService.init();

export { studyRecordService };