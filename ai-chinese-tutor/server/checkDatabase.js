const { sequelize, StudyRecord, Question } = require('./models');

async function checkDatabase() {
  try {
    console.log('=== 数据库检查开始 ===\n');

    // 检查连接
    await sequelize.authenticate();
    console.log('✓ 数据库连接成功\n');

    // 获取所有表
    const [results] = await sequelize.query("SELECT name FROM sqlite_master WHERE type='table'");
    console.log('数据库中的表：');
    results.forEach(row => console.log('  -', row.name));
    console.log();

    // 检查 StudyRecord 表
    console.log('StudyRecord 表统计：');
    const studyRecordCount = await StudyRecord.count();
    console.log('  总记录数:', studyRecordCount);

    if (studyRecordCount > 0) {
      const records = await StudyRecord.findAll({ limit: 3 });
      console.log('  最近3条记录：');
      records.forEach(r => {
        console.log(`    ID: ${r.id}, 用户ID: ${r.user_id}, 类型: ${r.type}, 分数: ${r.score}, 正确: ${r.is_correct}, 时间: ${r.createdAt}`);
      });
    }
    console.log();

    // 检查 Question 表
    console.log('Question 表统计：');
    const questionCount = await Question.count();
    console.log('  总题目数:', questionCount);

    const typeStats = await Question.findAll({
      attributes: [
        'type',
        [sequelize.fn('COUNT', '*'), 'count']
      ],
      group: ['type'],
      raw: true
    });

    console.log('  按类型统计：');
    typeStats.forEach(stat => {
      console.log(`    ${stat.type}: ${stat.count} 题`);
    });
    console.log();

    // 检查今日的学习记录
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const todayRecords = await StudyRecord.findAll({
      where: {
        createdAt: {
          [sequelize.Op.gte]: today
        }
      }
    });

    console.log('今日学习记录：');
    console.log('  记录数:', todayRecords.length);

    if (todayRecords.length > 0) {
      const todayScore = todayRecords.reduce((sum, r) => sum + (r.score || 0), 0);
      const todayCorrect = todayRecords.filter(r => r.is_correct === 1).length;

      console.log('  今日总得分:', todayScore);
      console.log('  今日正确题数:', todayCorrect);
      console.log('  今日正确率:', Math.round(todayCorrect / todayRecords.length * 100) + '%');
    }

    console.log('\n=== 数据库检查完成 ===');

  } catch (error) {
    console.error('错误:', error);
  } finally {
    await sequelize.close();
  }
}

checkDatabase();
