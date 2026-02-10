import api from './api';

class QuestionService {
  // 获取随机题目
  async getRandomQuestions(options = {}) {
    try {
      const params = {
        type: options.type,
        count: options.count || 10,
        difficulty: options.difficulty,
        year: options.year
      };

      const response = await api.get('/questions/random', { params });

      if (response.success) {
        // 转换数据格式以适配前端
        return response.data.map(q => this.formatQuestion(q));
      }

      return [];
    } catch (error) {
      console.error('获取随机题目失败:', error);
      return [];
    }
  }

  // 根据类型获取题目
  async getQuestionsByType(type, options = {}) {
    try {
      const params = {
        count: options.count || 10,
        difficulty: options.difficulty,
        year: options.year
      };

      const response = await api.get(`/questions/type/${type}`, { params });

      if (response.success) {
        return response.data.map(q => this.formatQuestion(q));
      }

      return [];
    } catch (error) {
      console.error('获取题目失败:', error);
      return [];
    }
  }

  // 根据年份获取题目
  async getQuestionsByYear(year, options = {}) {
    try {
      const params = {
        type: options.type,
        count: options.count || 20,
        difficulty: options.difficulty
      };

      const response = await api.get(`/questions/year/${year}`, { params });

      if (response.success) {
        return response.data.map(q => this.formatQuestion(q));
      }

      return [];
    } catch (error) {
      console.error('获取题目失败:', error);
      return [];
    }
  }

  // 格式化题目数据
  formatQuestion(question) {
    // 根据题型返回不同格式的数据
    switch (question.type) {
      case 'pinyin':
      case 'correction':
      case 'grammar':
        // 单选题
        return {
          id: question.id,
          type: 'single_choice',
          question: question.content,
          options: question.options ? Object.values(question.options) : [],
          correctAnswer: question.options ? this.getOptionIndex(question.options, question.answer) : 0,
          explanation: question.explanation,
          source: `${question.year}年${question.source || ''}`,
          difficulty: question.difficulty,
          year: question.year
        };

      case 'idiom':
      case 'vocabulary':
        // 可能是单选或多选
        const isMultiple = question.answer && question.answer.includes(/[、,]/);
        if (isMultiple) {
          // 多选题
          return {
            id: question.id,
            type: 'multiple_choice',
            question: question.content,
            options: question.options ? Object.values(question.options) : [],
            correctAnswer: question.answer ? this.getOptionIndexes(question.options, question.answer) : [],
            explanation: question.explanation,
            source: `${question.year}年${question.source || ''}`,
            difficulty: question.difficulty,
            year: question.year
          };
        } else {
          // 单选题
          return {
            id: question.id,
            type: 'single_choice',
            question: question.content,
            options: question.options ? Object.values(question.options) : [],
            correctAnswer: question.options ? this.getOptionIndex(question.options, question.answer) : 0,
            explanation: question.explanation,
            source: `${question.year}年${question.source || ''}`,
            difficulty: question.difficulty,
            year: question.year
          };
        }

      case 'literature':
        // 可能是填空题或单选题
        if (question.content.includes('__')) {
          // 填空题
          return {
            id: question.id,
            type: 'blank_filling',
            question: question.content,
            correctAnswer: question.answer,
            explanation: question.explanation,
            source: `${question.year}年${question.source || ''}`,
            difficulty: question.difficulty,
            year: question.year
          };
        } else {
          // 单选题
          return {
            id: question.id,
            type: 'single_choice',
            question: question.content,
            options: question.options ? Object.values(question.options) : [],
            correctAnswer: question.options ? this.getOptionIndex(question.options, question.answer) : 0,
            explanation: question.explanation,
            source: `${question.year}年${question.source || ''}`,
            difficulty: question.difficulty,
            year: question.year
          };
        }

      case 'reading':
        // 阅读理解（可能是单选或多选）
        if (question.answer && question.answer.length > 1) {
          // 可能是多选
          return {
            id: question.id,
            type: 'multiple_choice',
            question: question.content,
            options: question.options ? Object.values(question.options) : [],
            correctAnswer: this.getOptionIndexes(question.options, question.answer),
            explanation: question.explanation,
            source: `${question.year}年${question.source || ''}`,
            difficulty: question.difficulty,
            year: question.year
          };
        } else {
          return {
            id: question.id,
            type: 'single_choice',
            question: question.content,
            options: question.options ? Object.values(question.options) : [],
            correctAnswer: question.options ? this.getOptionIndex(question.options, question.answer) : 0,
            explanation: question.explanation,
            source: `${question.year}年${question.source || ''}`,
            difficulty: question.difficulty,
            year: question.year
          };
        }

      default:
        // 默认为单选题
        return {
          id: question.id,
          type: 'single_choice',
          question: question.content,
          options: question.options ? Object.values(question.options) : [],
          correctAnswer: question.options ? this.getOptionIndex(question.options, question.answer) : 0,
          explanation: question.explanation,
          source: `${question.year}年${question.source || ''}`,
          difficulty: question.difficulty,
          year: question.year
        };
    }
  }

  // 获取选项索引
  getOptionIndex(options, answer) {
    if (!options || !answer) return 0;
    const keys = Object.keys(options);
    const index = keys.findIndex(key => key === answer);
    return index >= 0 ? index : 0;
  }

  // 获取多个选项索引
  getOptionIndexes(options, answers) {
    if (!options || !answers) return [];
    const keys = Object.keys(options);
    const answerArray = answers.split(/[、,]/).map(a => a.trim());
    return answerArray.map(answer => {
      const index = keys.findIndex(key => key === answer);
      return index >= 0 ? index : 0;
    }).filter((value, index, array) => array.indexOf(value) === index); // 去重
  }

  // 搜索题目
  async searchQuestions(keyword, options = {}) {
    try {
      const params = {
        keyword,
        type: options.type,
        limit: options.limit || 20
      };

      const response = await api.get('/questions/search', { params });

      if (response.success) {
        return response.data.map(q => this.formatQuestion(q));
      }

      return [];
    } catch (error) {
      console.error('搜索题目失败:', error);
      return [];
    }
  }

  // 获取题目统计
  async getQuestionStats() {
    try {
      const response = await api.get('/questions/stats');

      if (response.success) {
        return response.data;
      }

      return null;
    } catch (error) {
      console.error('获取题目统计失败:', error);
      return null;
    }
  }
}

// 创建并导出服务实例
export const questionService = new QuestionService();

export default questionService;