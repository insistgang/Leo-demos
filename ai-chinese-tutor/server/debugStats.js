const { sequelize, StudyRecord } = require('./models');
const { Op } = require('sequelize');

async function debugStatistics() {
  try {
    console.log('=== 调试学习统计数据 ===\n');

    // 1. 查询总记录数
    const totalCount = await StudyRecord.count();
    console.log('1. 总学习记录数:', totalCount);

    // 2. 查询所有用户ID
    const [userIds] = await sequelize.query(`
      SELECT DISTINCT user_id, COUNT(*) as count
      FROM StudyRecord
      GROUP BY user_id
    `);
    console.log('\n2. 各用户的记录数:');
    userIds.forEach(u => {
      console.log(`  用户ID ${u.user_id}: ${u.count} 条记录`);
    });

    // 3. 查询最近的记录
    const recentRecords = await StudyRecord.findAll({
      limit: 5,
      order: [['createdAt', 'DESC']]
    });
    console.log('\n3. 最近5条记录:');
    recentRecords.forEach(r => {
      console.log(`  ID: ${r.id}, 用户: ${r.userId}, 类型: ${r.type}, 分数: ${r.score}, 时间: ${r.createdAt}`);
    });

    // 4. 模拟用户ID=2的统计查询
    console.log('\n4. 用户ID=2的统计:');
    const userId = 2;

    // 总记录
    const userTotalCount = await StudyRecord.count({
      where: { userId }
    });
    console.log(`  总记录数: ${userTotalCount}`);

    // 今日记录和得分
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const todayRecords = await StudyRecord.findAll({
      where: {
        userId,
        createdAt: { [Op.gte]: today }
      }
    });
    const todayScore = todayRecords.reduce((sum, r) => sum + (r.score || 0), 0);
    console.log(`  今日记录数: ${todayRecords.length}`);
    console.log(`  今日得分: ${todayScore}`);

    // 错题数量
    const wrongCount = await StudyRecord.count({
      where: {
        userId,
        score: { [Op.lt]: 10 }  // 假设满分10分，低于10分算错题
      }
    });
    console.log(`  错题数量: ${wrongCount}`);

    // 学习时长（假设每条记录平均30秒）
    const totalMinutes = Math.round(userTotalCount * 0.5);
    console.log(`  总学习时长: ${totalMinutes} 分钟`);

    // 学习天数
    const [studyDaysResult] = await sequelize.query(`
      SELECT COUNT(DISTINCT DATE(createdAt)) as days
      FROM StudyRecord
      WHERE userId = ${userId}
    `);
    console.log(`  学习天数: ${studyDaysResult[0].days}`);

  } catch (error) {
    console.error('错误:', error);
  } finally {
    await sequelize.close();
  }
}

debugStatistics();