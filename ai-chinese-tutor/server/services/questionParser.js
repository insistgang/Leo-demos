const fs = require('fs');
const path = require('path');
const { Sequelize } = require('sequelize');
const { Question } = require('../models');

class QuestionParser {
  constructor() {
    this.mdDir = path.join(__dirname, '../../k12_md');
    this.answersFile = path.join(__dirname, '../../k12_md/answers_2015_2024.txt');
    this.answers = new Map();
    this.loadAnswers();
  }

  // 加载答案到内存
  loadAnswers() {
    try {
      const content = fs.readFileSync(this.answersFile, 'utf-8');
      const lines = content.split('\n');

      lines.forEach(line => {
        const match = line.match(/^(\d{4})-(\d+):([A-D])$/);
        if (match) {
          const key = `${match[1]}-${match[2]}`;
          this.answers.set(key, match[3]);
        }
      });

      console.log(`已加载 ${this.answers.size} 个答案`);
    } catch (error) {
      console.error('加载答案文件失败:', error);
    }
  }

  // 解析所有 markdown 文件
  async parseAllFiles() {
    const files = fs.readdirSync(this.mdDir).filter(file => file.endsWith('.md'));
    const allQuestions = [];

    for (const file of files) {
      const year = parseInt(file.match(/\d{4}/)[0]);
      const filePath = path.join(this.mdDir, file);
      const content = fs.readFileSync(filePath, 'utf-8');

      console.log(`解析文件: ${file}`);
      const questions = this.parseMarkdown(content, year, file);
      allQuestions.push(...questions);
    }

    return allQuestions;
  }

  // 解析单个 markdown 文件
  parseMarkdown(content, year, filename) {
    const questions = [];
    const lines = content.split('\n');
    let currentQuestion = null;
    let questionNumber = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      // 跳过文件标题
      if (line.startsWith('# 20')) continue;

      // 识别题目开始
      const questionMatch = line.match(/^(\d+)\.\s+(.*)/);
      if (questionMatch) {
        // 保存上一个问题（如果有）
        if (currentQuestion) {
          // 尝试从答案文件获取答案
          const answerKey = `${year}-${currentQuestion.questionNumber}`;
          currentQuestion.answer = this.answers.get(answerKey) || '';
          questions.push(currentQuestion);
        }

        questionNumber = parseInt(questionMatch[1]);
        currentQuestion = {
          year,
          source: `${year}年体育单招文化考试`,
          questionNumber,
          content: questionMatch[2],
          options: {},
          answer: '',
          explanation: '',
          type: this.detectQuestionType(questionMatch[2]),
          difficulty: 3,
          tags: []
        };
        continue;
      }

      // 识别选项
      if (currentQuestion && /^[A-D]\./.test(line)) {
        const optionMatch = line.match(/^([A-D])\.\s*(.+)/);
        if (optionMatch) {
          currentQuestion.options[optionMatch[1]] = optionMatch[2];
        }
        continue;
      }

      // 识别解析
      if (currentQuestion && (line.includes('解析') || line.includes('考查') || line.includes('说明'))) {
        // 收集解析内容
        let explanation = [];
        for (let j = i; j < lines.length; j++) {
          if (lines[j].trim()) {
            explanation.push(lines[j].trim());
          }
          // 遇到下一个题目就停止
          if (lines[j].match(/^\d+\.\s+/)) {
            break;
          }
        }
        currentQuestion.explanation = explanation.join(' ');
        continue;
      }
    }

    // 保存最后一个问题
    if (currentQuestion) {
      // 尝试从答案文件获取答案
      const answerKey = `${year}-${currentQuestion.questionNumber}`;
      currentQuestion.answer = this.answers.get(answerKey) || '';
      questions.push(currentQuestion);
    }

    return questions;
  }

  // 检测题目类型
  detectQuestionType(content) {
    if (content.includes('读音') || content.includes('注音')) {
      return 'pinyin';
    }
    if (content.includes('错别字') || content.includes('错字')) {
      return 'correction';
    }
    if (content.includes('成语') || content.includes('熟语')) {
      return 'idiom';
    }
    if (content.includes('古诗') || content.includes('古文') || content.includes('古诗文')) {
      return 'literature';
    }
    if (content.includes('词语') || content.includes('词汇')) {
      return 'vocabulary';
    }
    if (content.includes('标点') || content.includes('病句')) {
      return 'grammar';
    }
    if (content.includes('阅读')) {
      return 'reading';
    }
    return 'comprehension';
  }

  // 批量导入题目到数据库
  async importQuestions() {
    try {
      console.log('开始解析真题文件...');
      const questions = await this.parseAllFiles();

      console.log(`共解析出 ${questions.length} 道题目`);

      // 批量插入数据库
      const insertedQuestions = await Question.bulkCreate(questions, {
        ignoreDuplicates: true
      });

      console.log(`成功导入 ${insertedQuestions.length} 道题目到数据库`);

      return {
        success: true,
        total: questions.length,
        inserted: insertedQuestions.length
      };
    } catch (error) {
      console.error('导入题目失败:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  // 根据条件获取随机题目
  async getRandomQuestions(options = {}) {
    const {
      type = null,
      count = 10,
      difficulty = null,
      year = null
    } = options;

    try {
      const whereCondition = {};

      if (type) whereCondition.type = type;
      if (difficulty) whereCondition.difficulty = difficulty;
      if (year) whereCondition.year = year;

      const questions = await Question.findAll({
        where: whereCondition,
        order: sequelize.literal('RANDOM()'),
        limit: count
      });

      return questions;
    } catch (error) {
      console.error('获取题目失败:', error);
      return [];
    }
  }
}

module.exports = QuestionParser;