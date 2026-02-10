// 词汇服务 - 提供词汇查询、管理等功能

class VocabularyService {
  constructor() {
    // 服务配置
    this.config = {
      apiProvider: 'local', // 'local' 或 'remote'
      timeout: 5000,
      cacheEnabled: true,
      cacheTTL: 3600000, // 1小时
    };
    
    // 初始化缓存
    this.cache = new Map();
  }
  
  // 初始化服务
  init() {
    console.log('VocabularyService initialized');
    // 可以在这里添加服务初始化逻辑
  }
  
  // 查询词语
  async searchWord(word) {
    try {
      // 检查缓存
      if (this.config.cacheEnabled) {
        const cachedResult = this.getFromCache(`word_${word}`);
        if (cachedResult) {
          return cachedResult;
        }
      }
      
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 300));
      
      // 模拟返回词语数据
      const mockResult = this.generateMockWordData(word);
      
      // 存入缓存
      if (this.config.cacheEnabled) {
        this.saveToCache(`word_${word}`, mockResult);
      }
      
      return mockResult;
    } catch (error) {
      console.error('查询词语失败:', error);
      throw new Error('查询词语失败，请稍后重试');
    }
  }
  
  // 本地查询（模拟）
  searchLocalWord(word) {
    // 模拟本地数据库查询
    const mockLocalWords = [
      { word: '你好', pinyin: 'nǐ hǎo', definition: '打招呼的用语' },
      { word: '学习', pinyin: 'xué xí', definition: '通过阅读、听讲、研究、实践等获得知识或技能' },
      { word: '语文', pinyin: 'yǔ wén', definition: '语言和文字的合称，特指中小学中的语言文字课' },
      { word: '体育', pinyin: 'tǐ yù', definition: '以发展体力、增强体质为主要任务的教育' },
      { word: '考试', pinyin: 'kǎo shì', definition: '考查知识或技能的一种方法' },
    ];
    
    return mockLocalWords.find(item => item.word === word);
  }
  
  // 获取相关词语（近义词、反义词等）
  async getRelatedWords(word) {
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 200));
      
      // 模拟返回相关词语数据
      const mockRelatedWords = {
        synonyms: this.getMockSynonyms(word),
        antonyms: this.getMockAntonyms(word),
        phrases: this.getMockPhrases(word),
        examples: this.getMockExamples(word)
      };
      
      return mockRelatedWords;
    } catch (error) {
      console.error('获取相关词语失败:', error);
      throw new Error('获取相关词语失败，请稍后重试');
    }
  }
  
  // 获取成语列表
  async getIdioms(page = 1, pageSize = 10) {
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 250));
      
      // 模拟成语数据
      const mockIdioms = [
        { id: 1, word: '一举两得', pinyin: 'yī jǔ liǎng dé', explanation: '做一件事得到两方面的好处' },
        { id: 2, word: '三心二意', pinyin: 'sān xīn èr yì', explanation: '又想这样又想那样，犹豫不定' },
        { id: 3, word: '四面八方', pinyin: 'sì miàn bā fāng', explanation: '各个方面或各个地方' },
        { id: 4, word: '五光十色', pinyin: 'wǔ guāng shí sè', explanation: '形容色泽鲜艳，花样繁多' },
        { id: 5, word: '六神无主', pinyin: 'liù shén wú zhǔ', explanation: '形容惊慌着急，没了主意，不知如何才好' },
        { id: 6, word: '七上八下', pinyin: 'qī shàng bā xià', explanation: '形容心里慌乱不安' },
        { id: 7, word: '九牛一毛', pinyin: 'jiǔ niú yī máo', explanation: '比喻极大数量中极微小的数量，微不足道' },
        { id: 8, word: '十全十美', pinyin: 'shí quán shí měi', explanation: '十分完美，毫无欠缺' },
        { id: 9, word: '百发百中', pinyin: 'bǎi fā bǎi zhòng', explanation: '形容射箭或打枪准确，每次都命中目标' },
        { id: 10, word: '千方百计', pinyin: 'qiān fāng bǎi jì', explanation: '想尽或用尽一切办法' },
      ];
      
      // 模拟分页
      const startIndex = (page - 1) * pageSize;
      const endIndex = startIndex + pageSize;
      const paginatedIdioms = mockIdioms.slice(startIndex, endIndex);
      
      return {
        data: paginatedIdioms,
        total: mockIdioms.length,
        page,
        pageSize
      };
    } catch (error) {
      console.error('获取成语列表失败:', error);
      throw new Error('获取成语列表失败，请稍后重试');
    }
  }
  
  // 获取练习题目
  async generatePracticeQuestions(type = 'mix', count = 5) {
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 400));
      
      // 模拟练习题目
      const mockQuestions = this.createMockPracticeQuestions(type, count);
      
      return mockQuestions;
    } catch (error) {
      console.error('生成练习题目失败:', error);
      throw new Error('生成练习题目失败，请稍后重试');
    }
  }
  
  // 收藏词语
  async favoriteWord(wordData) {
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 150));
      
      // 实际项目中这里应该调用后端API保存收藏
      // 这里只是模拟成功响应
      return {
        success: true,
        message: '收藏成功'
      };
    } catch (error) {
      console.error('收藏词语失败:', error);
      throw new Error('收藏词语失败，请稍后重试');
    }
  }
  
  // 取消收藏
  async unfavoriteWord(word) {
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 150));
      
      // 实际项目中这里应该调用后端API取消收藏
      // 这里只是模拟成功响应
      return {
        success: true,
        message: '已取消收藏'
      };
    } catch (error) {
      console.error('取消收藏失败:', error);
      throw new Error('取消收藏失败，请稍后重试');
    }
  }
  
  // 获取收藏列表
  async getFavoriteWords(page = 1, pageSize = 20) {
    try {
      // 模拟API调用延迟
      await new Promise(resolve => setTimeout(resolve, 200));
      
      // 模拟从本地存储获取收藏词语
      const favorites = localStorage.getItem('favoriteWords');
      const favoriteList = favorites ? JSON.parse(favorites) : [];
      
      // 模拟分页
      const startIndex = (page - 1) * pageSize;
      const endIndex = startIndex + pageSize;
      const paginatedFavorites = favoriteList.slice(startIndex, endIndex);
      
      return {
        data: paginatedFavorites,
        total: favoriteList.length,
        page,
        pageSize
      };
    } catch (error) {
      console.error('获取收藏列表失败:', error);
      throw new Error('获取收藏列表失败，请稍后重试');
    }
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
  
  saveToCache(key, data) {
    const expiry = Date.now() + this.config.cacheTTL;
    this.cache.set(key, { data, expiry });
  }
  
  // 生成模拟词语数据
  generateMockWordData(word) {
    // 根据不同的词生成不同的模拟数据
    const mockTemplates = {
      '学习': {
        word: '学习',
        pinyin: 'xué xí',
        pronunciation: `https://example.com/pronunciation/${word}.mp3`,
        definitions: [
          '通过阅读、听讲、研究、实践等获得知识或技能。',
          '效法；模仿。'
        ],
        partOfSpeech: '动词',
        radical: '子',
        strokeCount: 14,
        examples: [
          '我们要努力学习科学文化知识。',
          '他很善于学习别人的优点。'
        ],
        origin: '《论语·季氏》："不学礼，无以立。"'
      },
      '语文': {
        word: '语文',
        pinyin: 'yǔ wén',
        pronunciation: `https://example.com/pronunciation/${word}.mp3`,
        definitions: [
          '语言和文字的合称。',
          '特指中小学中的语言文字课。'
        ],
        partOfSpeech: '名词',
        radical: '讠/文',
        strokeCount: 12,
        examples: [
          '语文是一门基础学科。',
          '他的语文成绩很好。'
        ],
        origin: '《红楼梦》第三回："黛玉入学名册，上写着：金陵十二钗正册之林黛玉。"'
      }
    };
    
    // 如果有特定模板就使用，否则使用通用模板
    if (mockTemplates[word]) {
      return mockTemplates[word];
    }
    
    // 通用模板
    return {
      word: word,
      pinyin: 'mó nǐ pīn yīn',
      pronunciation: `https://example.com/pronunciation/${word}.mp3`,
      definitions: [
        '这是一个模拟的词语释义。',
        '在实际应用中，这里会显示真实的词语解释。'
      ],
      partOfSpeech: '名词',
      radical: '模',
      strokeCount: 10,
      examples: [
        `这是一个包含"${word}"的例句。`,
        `在实际应用中，这里会显示包含"${word}"的真实例句。`
      ],
      origin: '出自《模拟词典》'
    };
  }
  
  // 获取模拟近义词
  getMockSynonyms(word) {
    const synonymMap = {
      '学习': ['研习', '进修', '求学', '练习'],
      '语文': ['国文', '中文', '语言', '文字'],
      '体育': ['运动', '体能', '健身', '锻炼'],
      '考试': ['测验', '考查', '考核', '测试']
    };
    
    return synonymMap[word] || ['相近', '类似', '相似', '相仿'];
  }
  
  // 获取模拟反义词
  getMockAntonyms(word) {
    const antonymMap = {
      '学习': ['贪玩', '厌学', '放弃', '颓废'],
      '语文': ['外语', '外文', '方言', '土语'],
      '体育': ['文弱', '娇气', '柔弱', '虚弱'],
      '考试': ['免试', '免考', '豁免', '弃权']
    };
    
    return antonymMap[word] || ['相反', '对立', '矛盾', '不同'];
  }
  
  // 获取模拟词组
  getMockPhrases(word) {
    const phraseMap = {
      '学习': ['学习成绩', '学习态度', '学习方法', '学习效率'],
      '语文': ['语文课本', '语文教师', '语文考试', '语文水平'],
      '体育': ['体育锻炼', '体育比赛', '体育老师', '体育场馆'],
      '考试': ['考试成绩', '考试作弊', '考试压力', '考试大纲']
    };
    
    return phraseMap[word] || [`${word}相关`, `${word}知识`, `${word}技能`, `${word}应用`];
  }
  
  // 获取模拟例句
  getMockExamples(word) {
    const exampleMap = {
      '学习': [
        '我们要好好学习，天天向上。',
        '学习是一种终身的事业。',
        '通过不断学习，他取得了很大的进步。'
      ],
      '语文': [
        '学好语文是学习其他学科的基础。',
        '语文能力的提高需要长期积累。',
        '他的语文素养很高，写得一手好文章。'
      ],
      '体育': [
        '体育锻炼可以增强体质。',
        '我们学校很重视体育教学。',
        '他是体育特长生，擅长跑步。'
      ],
      '考试': [
        '明天就要考试了，我要好好复习。',
        '考试成绩不能代表一切。',
        '他在考试中发挥得很好。'
      ]
    };
    
    if (exampleMap[word]) {
      return exampleMap[word];
    }
    
    // 生成通用例句
    return [
      `这个句子包含了"${word}"这个词。`,
      `"${word}"在这句话中是什么意思？`,
      `请用"${word}"造一个句子。`
    ];
  }
  
  // 创建模拟练习题目
  createMockPracticeQuestions(type, count) {
    const questions = [];
    let questionId = 1;
    
    // 根据类型生成不同的题目
    if (type === 'mix' || type === 'definition') {
      // 定义理解题
      questions.push({
        id: questionId++, 
        type: 'definition',
        question: '"一举两得"的意思是什么？',
        options: [
          '做一件事得到两方面的好处',
          '做两件事得到一方面的好处',
          '做很多事得到很多好处',
          '什么好处都没有'
        ],
        correctAnswer: 0
      });
      
      if (count > 1) {
        questions.push({
          id: questionId++, 
          type: 'definition',
          question: '"三心二意"的意思是什么？',
          options: [
            '意志坚定，目标明确',
            '又想这样又想那样，犹豫不定',
            '有三个想法，两个心意',
            '形容非常专心'
          ],
          correctAnswer: 1
        });
      }
    }
    
    if (type === 'mix' || type === 'synonym') {
      // 近义词题
      questions.push({
        id: questionId++, 
        type: 'synonym',
        question: '"学习"的近义词是什么？',
        options: ['创新', '创造', '研习', '发明'],
        correctAnswer: 2
      });
    }
    
    if (type === 'mix' || type === 'antonym') {
      // 反义词题
      questions.push({
        id: questionId++, 
        type: 'antonym',
        question: '"大"的反义词是什么？',
        options: ['多', '小', '高', '胖'],
        correctAnswer: 1
      });
    }
    
    if (type === 'mix' || type === 'usage') {
      // 用法题
      questions.push({
        id: questionId++, 
        type: 'usage',
        question: '下面哪个句子中"语文"的用法是正确的？',
        options: [
          '我很喜欢语文这门学科。',
          '今天的天气很语文。',
          '他的语文非常高。',
          '请把这个语文打开。'
        ],
        correctAnswer: 0
      });
    }
    
    // 如果需要更多题目，复制现有题目并修改
    while (questions.length < count) {
      const randomIndex = Math.floor(Math.random() * questions.length);
      const clonedQuestion = JSON.parse(JSON.stringify(questions[randomIndex]));
      clonedQuestion.id = questionId++;
      questions.push(clonedQuestion);
    }
    
    // 限制题目数量
    return questions.slice(0, count);
  }
}

// 创建单例实例
const vocabularyService = new VocabularyService();

// 初始化服务
vocabularyService.init();

export { vocabularyService };