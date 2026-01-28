const { Question, sequelize } = require('../models');

class QuestionGenerator {
  constructor() {
    // 题目模板库
    this.templates = {
      pinyin: [
        {
          pattern: "下列词语中加点字的读音全都正确的一项是",
          options: ["A. {word1}（{pinyin1}）  {word2}（{pinyin2}）",
                   "B. {word1}（{wrongPinyin1}）  {word2}（{pinyin2}）",
                   "C. {word1}（{pinyin1}）  {word2}（{wrongPinyin2}）",
                   "D. {word1}（{wrongPinyin1}）  {word2}（{wrongPinyin2}）"]
        },
        {
          pattern: "给下列句子中的加点字选择正确的读音",
          context: "句子：{sentence}"
        }
      ],

      vocabulary: [
        {
          pattern: "下列句子中加点成语使用正确的一项是",
          options: ["A. {sentence1}", "B. {sentence2}", "C. {sentence3}", "D. {sentence4}"]
        },
        {
          pattern: "依次填入下列各句横线处的词语，最恰当的一项是",
          context: "（1）{context1}（2）{context2}"
        }
      ],

      literature: [
        {
          pattern: "补全下列诗句",
          context: "{poemFront}，________。"
        },
        {
          pattern: "对下列诗句的赏析，不正确的一项是",
          context: "诗句：{poem}"
        }
      ],

      idiom: [
        {
          pattern: "下列成语的解释，正确的一项是",
          options: ["A. {idiom1}：{explanation1}", "B. {idiom2}：{explanation2}",
                   "C. {idiom3}：{explanation3}", "D. {idiom4}：{explanation4}"]
        },
        {
          pattern: "填入下列句子中的成语，最恰当的一项是",
          context: "{sentence}"
        }
      ]
    };

    // 词语库（从真题中提取）
    this.wordBank = {
      common: ["差强人意", "首当其冲", "脱颖而出", "相得益彰", "蔚然成风",
               "日新月异", "博大精深", "源远流长", "艰苦奋斗", "自强不息"],
      pinyin: [
        {word: "媲美", pinyin: "pì", wrong: "bǐ"},
        {word: "载重", pinyin: "zài", wrong: "zǎi"},
        {word: "参差", pinyin: "cēncī", wrong: "cānchā"},
        {word: "炽热", pinyin: "chì", wrong: "zhì"},
        {word: "逮捕", pinyin: "dài", wrong: "dǎi"}
      ]
    };

    // 古诗词库
    this.poetryBank = [
      {front: "床前明月光，疑是地上霜", back: "举头望明月，低头思故乡", author: "李白", title: "静夜思"},
      {front: "春眠不觉晓", back: "处处闻啼鸟", author: "孟浩然", title: "春晓"},
      {front: "白日依山尽", back: "黄河入海流", author: "王之涣", title: "登鹳雀楼"},
      {front: "千山鸟飞绝", back: "万径人踪灭", author: "柳宗元", title: "江雪"},
      {front: "独在异乡为异客", back: "每逢佳节倍思亲", author: "王维", title: "九月九日忆山东兄弟"}
    ];
  }

  // 生成类似题目
  async generateSimilarQuestion(type, count = 5) {
    const similarQuestions = [];

    // 获取该类型的现有题目作为参考
    const existingQuestions = await Question.findAll({
      where: { type },
      limit: 10,
      order: sequelize.literal('RANDOM()')
    });

    for (let i = 0; i < count; i++) {
      const question = this.generateQuestion(type, existingQuestions);
      if (question) {
        similarQuestions.push(question);
      }
    }

    return similarQuestions;
  }

  // 生成单个题目
  generateQuestion(type, references = []) {
    const templates = this.templates[type];
    if (!templates || templates.length === 0) {
      return null;
    }

    // 随机选择一个模板
    const template = templates[Math.floor(Math.random() * templates.length)];

    switch (type) {
      case 'pinyin':
        return this.generatePinyinQuestion(template, references);
      case 'vocabulary':
        return this.generateVocabularyQuestion(template, references);
      case 'literature':
        return this.generateLiteratureQuestion(template, references);
      case 'idiom':
        return this.generateIdiomQuestion(template, references);
      default:
        return this.generateGenericQuestion(type, template, references);
    }
  }

  // 生成拼音题
  generatePinyinQuestion(template, references) {
    const words = this.getRandomItems(this.wordBank.pinyin, 2);
    const correctAnswer = Math.floor(Math.random() * 4);

    const options = [
      `A. ${words[0].word}（${words[0].pinyin}）  ${words[1].word}（${words[1].pinyin}）`,
      `B. ${words[0].word}（${words[0].wrong}）  ${words[1].word}（${words[1].pinyin}）`,
      `C. ${words[0].word}（${words[0].pinyin}）  ${words[1].word}（${words[1].wrong}）`,
      `D. ${words[0].word}（${words[0].wrong}）  ${words[1].word}（${words[1].wrong}）`
    ];

    return {
      type: 'pinyin',
      content: template.pattern,
      options: {
        A: options[0].substring(3),
        B: options[1].substring(3),
        C: options[2].substring(3),
        D: options[3].substring(3)
      },
      answer: String.fromCharCode(65 + correctAnswer), // A, B, C, D
      explanation: `这道题考查汉字的读音。${words[0].word}的正确读音是${words[0].pinyin}，${words[1].word}的正确读音是${words[1].pinyin}。`,
      difficulty: '中等',
      source: '练习题',
      year: new Date().getFullYear(),
      isGenerated: true
    };
  }

  // 生成词汇题
  generateVocabularyQuestion(template, references) {
    const idioms = this.getRandomItems(this.wordBank.common, 4);
    const correctAnswer = Math.floor(Math.random() * 4);

    // 生成句子
    const sentences = idioms.map((idiom, index) => {
      const isCorrect = index === correctAnswer;
      const templates = [
        `他的表现${isCorrect ? '' : '不'}${idiom}，还需要继续努力。`,
        `这次活动${idiom}，大家都非常满意。`,
        `在学习上，我们应该${idiom}，永不放弃。`,
        `他的进步${idiom}，老师都看在眼里。`
      ];
      return templates[index % templates.length];
    });

    return {
      type: 'vocabulary',
      content: template.pattern,
      options: {
        A: sentences[0],
        B: sentences[1],
        C: sentences[2],
        D: sentences[3]
      },
      answer: String.fromCharCode(65 + correctAnswer),
      explanation: `这道题考查成语的使用。${idioms[correctAnswer]}在句子中使用正确。`,
      difficulty: '中等',
      source: '练习题',
      year: new Date().getFullYear(),
      isGenerated: true
    };
  }

  // 生成文学题
  generateLiteratureQuestion(template, references) {
    const poem = this.getRandomItem(this.poetryBank);
    const isBlankFilling = template.pattern.includes('________');

    if (isBlankFilling) {
      return {
        type: 'literature',
        content: `${poem.front}，________。`,
        answer: poem.back,
        explanation: `这句诗出自${poem.author}的《${poem.title}》`,
        difficulty: '简单',
        source: '古诗练习',
        year: new Date().getFullYear(),
        isGenerated: true
      };
    }

    // 生成选择题
    const options = [
      `A. 这句诗表达了诗人对故乡的思念`,
      `B. 这句诗描绘了春天的美景`,
      `C. 这句诗抒发了诗人的豪情壮志`,
      `D. 这句诗反映了诗人的人生感悟`
    ];
    const correctAnswer = Math.floor(Math.random() * 4);

    return {
      type: 'literature',
      content: `对下列诗句的赏析，正确的一项是\n${poem.front}，${poem.back}`,
      options: {
        A: options[0].substring(3),
        B: options[1].substring(3),
        C: options[2].substring(3),
        D: options[3].substring(3)
      },
      answer: String.fromCharCode(65 + correctAnswer),
      explanation: `这句诗出自${poem.author}的《${poem.title}》，表达了深刻的情感。`,
      difficulty: '中等',
      source: '古诗练习',
      year: new Date().getFullYear(),
      isGenerated: true
    };
  }

  // 生成成语题
  generateIdiomQuestion(template, references) {
    const idioms = this.getRandomItems(this.wordBank.common, 4);
    const correctAnswer = Math.floor(Math.random() * 4);

    // 生成解释
    const explanations = [
      "形容大体上还能使人满意",
      "表示最先受到攻击或遭遇灾难",
      "比喻才能全部显示出来",
      "指互相配合、补充，更能显出好处"
    ];

    return {
      type: 'idiom',
      content: template.pattern,
      options: {
        A: `${idioms[0]}：${explanations[0]}`,
        B: `${idioms[1]}：${explanations[1]}`,
        C: `${idioms[2]}：${explanations[2]}`,
        D: `${idioms[3]}：${explanations[3]}`
      },
      answer: String.fromCharCode(65 + correctAnswer),
      explanation: `${idioms[correctAnswer]}的正确解释是：${explanations[correctAnswer]}`,
      difficulty: '中等',
      source: '成语练习',
      year: new Date().getFullYear(),
      isGenerated: true
    };
  }

  // 生成通用题目
  generateGenericQuestion(type, template, references) {
    const ref = references[0];
    if (!ref) return null;

    // 基于参考题目生成变体
    return {
      type: type,
      content: ref.content.replace(/[0-9]{4}/g, String(new Date().getFullYear())),
      options: ref.options,
      answer: ref.answer,
      explanation: `这是基于真题生成的练习题。${ref.explanation}`,
      difficulty: ref.difficulty || '中等',
      source: '练习题',
      year: new Date().getFullYear(),
      isGenerated: true
    };
  }

  // 保存生成的题目到数据库
  async saveGeneratedQuestions(questions) {
    const savedQuestions = [];

    for (const question of questions) {
      try {
        const saved = await Question.create({
          ...question,
          questionNumber: await this.getNextQuestionNumber(question.type, question.year)
        });
        savedQuestions.push(saved);
      } catch (error) {
        console.error('保存生成的题目失败:', error);
      }
    }

    return savedQuestions;
  }

  // 获取下一个题目编号
  async getNextQuestionNumber(type, year) {
    const maxQuestion = await Question.findOne({
      where: { type, year },
      order: [['questionNumber', 'DESC']]
    });

    return maxQuestion ? maxQuestion.questionNumber + 1 : 1;
  }

  // 工具函数：获取随机项
  getRandomItem(array) {
    return array[Math.floor(Math.random() * array.length)];
  }

  // 工具函数：获取多个随机项
  getRandomItems(array, count) {
    const shuffled = [...array].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
  }

  // 为特定类型生成题目
  async generateAndSave(type, count = 10) {
    console.log(`正在生成 ${count} 道${type}类型的练习题...`);

    const questions = await this.generateSimilarQuestion(type, count);
    const savedQuestions = await this.saveGeneratedQuestions(questions);

    console.log(`成功生成并保存了 ${savedQuestions.length} 道题目`);
    return savedQuestions;
  }
}

module.exports = QuestionGenerator;