const { StudyRecord } = require('./models');

async function cleanInvalidRecords() {
  try {
    console.log('=== 清理无效的学习记录 ===\n');

    // 查找所有 user_id 为 null 的记录
    const invalidRecords = await StudyRecord.findAll({
      where: {
        userId: null
      }
    });

    console.log(`找到 ${invalidRecords.length} 条无效记录（user_id 为 null）`);

    if (invalidRecords.length > 0) {
      // 删除这些记录
      const deletedCount = await StudyRecord.destroy({
        where: {
          userId: null
        }
      });

      console.log(`✓ 成功删除 ${deletedCount} 条无效记录`);
    }

    // 显示剩余的记录统计
    const totalRecords = await StudyRecord.count();
    console.log(`\n剩余有效记录数: ${totalRecords}`);

    // 按用户统计
    const userStats = await StudyRecord.findAll({
      attributes: [
        'userId',
        [require('sequelize').fn('COUNT', '*'), 'count']
      ],
      group: ['userId'],
      raw: true
    });

    console.log('\n按用户统计:');
    userStats.forEach(stat => {
      console.log(`  用户ID ${stat.userId}: ${stat.count} 条记录`);
    });

    console.log('\n=== 清理完成 ===');

  } catch (error) {
    console.error('错误:', error);
  } finally {
    require('./models').sequelize.close();
  }
}

cleanInvalidRecords();