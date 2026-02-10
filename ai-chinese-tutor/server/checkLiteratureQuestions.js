const { Question } = require('./models');

async function checkLiteratureQuestions() {
  try {
    console.log('=== 检查古诗词题目的答案格式 ===\n');

    // 查找古诗词题目
    const questions = await Question.findAll({
      where: { type: 'literature' },
      limit: 5
    });

    questions.forEach((q, index) => {
      console.log(`题目 ${index + 1}:`);
      console.log('  ID:', q.id);
      console.log('  题目:', q.content?.question || '无');
      console.log('  选项:', q.content?.options || '无');
      console.log('  答案:', q.answer);
      console.log('  答案类型:', typeof q.answer);
      console.log('  ---');
    });

  } catch (error) {
    console.error('错误:', error);
  } finally {
    require('./models').sequelize.close();
  }
}

checkLiteratureQuestions();