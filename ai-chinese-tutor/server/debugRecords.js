const { sequelize, StudyRecord, Question } = require('./models');

async function debugRecords() {
  try {
    console.log('调试学习记录...\n');

    // 获取最近的学习记录
    const records = await StudyRecord.findAll({
      limit: 5,
      order: [['createdAt', 'DESC']]
    });

    console.log('最近5条学习记录详情：');
    for (const record of records) {
      console.log('\n-------------------');
      console.log('记录ID:', record.id);
      console.log('用户ID:', record.userId);
      console.log('类型:', record.type);
      console.log('得分:', record.score);
      console.log('正确答案数:', record.correctAnswers);
      console.log('总题数:', record.totalQuestions);
      console.log('内容:', JSON.stringify(record.content, null, 2));

      // 如果有题目ID，查看题目详情
      if (record.content && record.content.questionId) {
        const question = await Question.findByPk(record.content.questionId);
        if (question) {
          console.log('题目答案:', question.answer);
          console.log('用户答案:', record.content.selectedAnswer);
          console.log('是否匹配:', record.content.selectedAnswer === question.answer ? '是' : '否');
        }
      }
    }

  } catch (error) {
    console.error('调试失败:', error);
  } finally {
    await sequelize.close();
  }
}

// 运行调试
debugRecords();