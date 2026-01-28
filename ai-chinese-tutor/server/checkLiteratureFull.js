const { Question } = require('./models');

async function checkLiteratureFull() {
  try {
    console.log('=== 检查古诗词题目完整数据结构 ===\n');

    // 查找一道古诗词题
    const question = await Question.findOne({
      where: { type: 'literature' }
    });

    if (question) {
      console.log('题目完整数据:');
      console.log(JSON.stringify(question.toJSON(), null, 2));
    }

  } catch (error) {
    console.error('错误:', error);
  } finally {
    require('./models').sequelize.close();
  }
}

checkLiteratureFull();