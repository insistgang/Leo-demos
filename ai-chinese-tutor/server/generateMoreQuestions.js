const QuestionGenerator = require('./services/questionGenerator');
const { sequelize } = require('./models');

async function generateQuestions() {
  console.log('开始生成练习题...\n');

  try {
    // 确保数据库连接
    await sequelize.authenticate();
    console.log('数据库连接成功');

    const generator = new QuestionGenerator();

    // 各类型题目生成数量
    const questionTypes = [
      { type: 'pinyin', count: 20 },
      { type: 'vocabulary', count: 20 },
      { type: 'literature', count: 20 },
      { type: 'idiom', count: 20 },
      { type: 'grammar', count: 15 },
      { type: 'correction', count: 15 }
    ];

    let totalGenerated = 0;

    // 为每种类型生成题目
    for (const { type, count } of questionTypes) {
      console.log(`\n=== 生成 ${type} 类型题目 ===`);
      const savedQuestions = await generator.generateAndSave(type, count);
      totalGenerated += savedQuestions.length;

      // 显示生成的题目示例
      if (savedQuestions.length > 0) {
        console.log('\n示例题目:');
        const example = savedQuestions[0];
        console.log('题目:', example.content.substring(0, 100) + '...');
        console.log('答案:', example.answer);
        console.log('类型:', example.type);
      }
    }

    console.log('\n' + '='.repeat(50));
    console.log(`总共生成了 ${totalGenerated} 道练习题！`);
    console.log('='.repeat(50));

    // 更新统计
    const { Question } = require('./models');
    const stats = await Question.findAll({
      attributes: [
        'type',
        [require('sequelize').fn('COUNT', '*'), 'count']
      ],
      group: ['type']
    });

    console.log('\n更新后的题目统计:');
    stats.forEach(s => {
      console.log(`${s.type}: ${s.dataValues.count}道`);
    });

    const total = await Question.count();
    console.log(`\n总题目数: ${total}道`);

  } catch (error) {
    console.error('生成题目时出错:', error);
  } finally {
    await sequelize.close();
  }
}

// 运行生成脚本
if (require.main === module) {
  generateQuestions();
}

module.exports = generateQuestions;