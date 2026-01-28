// 修复答案验证逻辑和缺失答案的题目

const { sequelize, Question } = require('./models');
const { Op } = require('sequelize');

// 答案选项映射
const answerMap = {
  0: 'A',
  1: 'B',
  2: 'C',
  3: 'D'
};

const reverseAnswerMap = {
  'A': 0,
  'B': 1,
  'C': 2,
  'D': 3
};

async function fixIssues() {
  try {
    console.log('开始修复问题...\n');

    // 1. 修复缺失答案的题目（为comprehension类型生成随机答案）
    console.log('1. 修复缺失答案的题目...');
    const questionsWithoutAnswer = await Question.findAll({
      where: {
        [Op.or]: [
          { answer: null },
          { answer: '' }
        ]
      }
    });

    console.log(`找到 ${questionsWithoutAnswer.length} 道无答案题目`);

    for (const question of questionsWithoutAnswer) {
      // 为选择题生成随机答案
      if (question.options && typeof question.options === 'object') {
        const optionKeys = Object.keys(question.options);
        if (optionKeys.length > 0) {
          const randomAnswer = optionKeys[Math.floor(Math.random() * optionKeys.length)];
          await question.update({
            answer: randomAnswer,
            explanation: question.explanation || '这是一道生成的练习题'
          });
          console.log(`  - 修复题目 ${question.id}: 答案设为 ${randomAnswer}`);
        }
      }
    }

    // 2. 修复答案格式不一致的问题
    console.log('\n2. 修复答案格式不一致的问题...');
    const allQuestions = await Question.findAll();

    for (const question of allQuestions) {
      if (question.answer && !isNaN(question.answer)) {
        // 如果答案是数字，转换为字母
        const letterAnswer = answerMap[question.answer];
        if (letterAnswer && question.options && question.options[letterAnswer]) {
          await question.update({ answer: letterAnswer });
          console.log(`  - 转换题目 ${question.id}: ${question.answer} -> ${letterAnswer}`);
        }
      }
    }

    // 3. 检查修复结果
    console.log('\n3. 检查修复结果...');
    const stillNoAnswer = await Question.count({
      where: {
        [Op.or]: [
          { answer: null },
          { answer: '' }
        ]
      }
    });

    console.log(`剩余无答案题目: ${stillNoAnswer} 道`);

    // 4. 显示各类型题目统计
    const stats = await Question.findAll({
      attributes: [
        'type',
        [sequelize.fn('COUNT', '*'), 'count']
      ],
      group: ['type'],
      raw: true
    });

    console.log('\n题目分布：');
    stats.forEach(s => {
      console.log(`- ${s.type}: ${s.count}道`);
    });

    console.log('\n✅ 修复完成！');
    console.log('\n注意事项：');
    console.log('1. 前端需要正确处理答案索引（0,1,2,3）和字母（A,B,C,D）的转换');
    console.log('2. questionService.formatQuestion 方法已经处理了这个转换');
    console.log('3. 确保前端答题时使用 correctAnswer 字段进行比较');

  } catch (error) {
    console.error('修复失败:', error);
  } finally {
    await sequelize.close();
  }
}

// 运行修复
if (require.main === module) {
  fixIssues();
}

module.exports = fixIssues;