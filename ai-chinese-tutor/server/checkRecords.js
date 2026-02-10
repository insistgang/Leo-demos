const { sequelize } = require('./models');

async function checkRecords() {
  try {
    console.log('=== 检查学习记录的 user_id ===\n');

    // 直接查询原始数据
    const [results] = await sequelize.query(`
      SELECT id, user_id, type, score, createdAt
      FROM StudyRecord
      ORDER BY id DESC
      LIMIT 10
    `);

    console.log('最近10条记录:');
    results.forEach(row => {
      console.log(`  ID: ${row.id}, user_id: ${row.user_id}, 类型: ${row.type}, 分数: ${row.score}, 时间: ${row.created_at}`);
    });

    // 检查是否有 NULL 或空的 user_id
    const [invalidResults] = await sequelize.query(`
      SELECT COUNT(*) as count
      FROM StudyRecord
      WHERE user_id IS NULL OR user_id = '' OR user_id = 0
    `);

    console.log(`\n无效 user_id 的记录数: ${invalidResults[0].count}`);

    // 检查具体哪些记录的 user_id 是 NULL
    if (invalidResults[0].count > 0) {
      const [invalidRows] = await sequelize.query(`
        SELECT id, user_id, type, score
        FROM StudyRecord
        WHERE user_id IS NULL OR user_id = '' OR user_id = 0
        LIMIT 5
      `);

      console.log('\n无效记录示例:');
      invalidRows.forEach(row => {
        console.log(`  ID: ${row.id}, user_id: ${row.user_id}, 类型: ${row.type}`);
      });
    }

  } catch (error) {
    console.error('错误:', error);
  } finally {
    await sequelize.close();
  }
}

checkRecords();