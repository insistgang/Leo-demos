const { StudyRecord } = require('./models');

async function checkStudyRecords() {
  try {
    console.log('连接数据库...');

    // 检查 StudyRecord 表是否存在
    const tableExists = await StudyRecord.sequelize.getQueryInterface().showAllTables()
      .then(tables => tables.includes('StudyRecords'));

    console.log('StudyRecord 表存在:', tableExists);

    if (!tableExists) {
      console.log('表不存在，需要同步数据库');
      await StudyRecord.sequelize.sync();
      return;
    }

    // 查询记录总数
    const totalCount = await StudyRecord.count();
    console.log('\nStudyRecord 总记录数:', totalCount);

    // 查询最近10条记录
    const recentRecords = await StudyRecord.findAll({
      order: [['createdAt', 'DESC']],
      limit: 10,
      raw: true
    });

    console.log('\n最近10条记录:');
    if (recentRecords.length === 0) {
      console.log('（无记录）');
    } else {
      recentRecords.forEach((r, i) => {
        console.log(`\n记录 ${i + 1}:`);
        console.log(`  ID: ${r.id}`);
        console.log(`  用户ID: ${r.userId}`);
        console.log(`  类型: ${r.type}`);
        console.log(`  题目ID: ${r.questionId}`);
        console.log(`  得分: ${r.score}`);
        console.log(`  是否正确: ${r.correctAnswers}`);
        console.log(`  创建时间: ${r.createdAt}`);
      });
    }

    // 按用户统计
    const userStats = await StudyRecord.findAll({
      attributes: [
        'userId',
        [require('sequelize').fn('COUNT', '*'), 'count'],
        [require('sequelize').fn('SUM', require('sequelize').col('score')), 'totalScore']
      ],
      group: ['userId'],
      raw: true
    });

    console.log('\n用户统计:');
    userStats.forEach(stat => {
      console.log(`  用户${stat.userId}: ${stat.count}条记录，总分${stat.totalScore}`);
    });

    await StudyRecord.sequelize.close();

  } catch (error) {
    console.error('检查失败:', error);
  }
}

checkStudyRecords();