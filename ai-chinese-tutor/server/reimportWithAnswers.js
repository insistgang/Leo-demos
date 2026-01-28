const { Sequelize, DataTypes } = require('sequelize');
const Question = require('./models');

// 创建数据库连接
const sequelize = new Sequelize({
  dialect: 'sqlite',
  storage: './database.db',
  logging: console.log
});

async function reimportWithAnswers() {
  try {
    console.log('连接数据库...');
    await sequelize.authenticate();
    console.log('数据库连接成功');

    // 同步数据库模型
    console.log('同步数据库模型...');
    await sequelize.sync({ force: false });
    console.log('数据库同步完成');

    // 删除所有现有题目
    console.log('删除现有题目...');
    const QuestionModel = sequelize.models.Question;
    await QuestionModel.destroy({
      where: {},
      truncate: true
    });
    console.log('已删除所有现有题目');

    // 重新导入题目（包含答案）
    console.log('开始重新导入题目...');
    const QuestionParser = require('./services/questionParser');
    const parser = new QuestionParser();

    const questions = await parser.parseAllFiles();
    console.log(`解析出 ${questions.length} 道题目`);

    // 批量插入
    const insertedQuestions = await QuestionModel.bulkCreate(questions);
    console.log(`成功导入 ${insertedQuestions.length} 道题目`);

    // 验证导入结果
    const sampleQuestions = await QuestionModel.findAll({
      where: { type: 'pinyin' },
      limit: 5
    });

    console.log('\n验证导入的拼音题目：');
    sampleQuestions.forEach((q, index) => {
      console.log(`\n题目${index + 1}:`);
      console.log('内容:', q.content);
      console.log('答案:', q.answer);
    });

    await sequelize.close();

  } catch (error) {
    console.error('重新导入失败:', error);
  }
}

reimportWithAnswers();