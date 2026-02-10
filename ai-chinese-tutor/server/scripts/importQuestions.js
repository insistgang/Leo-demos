require('dotenv').config();
const { sequelize } = require('../models');
const QuestionParser = require('../services/questionParser');

async function importQuestions() {
  try {
    console.log('连接数据库...');
    await sequelize.authenticate();
    console.log('数据库连接成功');

    console.log('同步数据库模型...');
    await sequelize.sync({ force: false });
    console.log('数据库同步完成');

    const parser = new QuestionParser();
    const result = await parser.importQuestions();

    if (result.success) {
      console.log('\n✅ 导入成功！');
      console.log(`总计题目: ${result.total}`);
      console.log(`成功导入: ${result.inserted}`);
    } else {
      console.log('\n❌ 导入失败！');
      console.error(result.error);
    }

    process.exit(0);
  } catch (error) {
    console.error('\n导入过程中出错:', error);
    process.exit(1);
  }
}

// 运行导入
importQuestions();