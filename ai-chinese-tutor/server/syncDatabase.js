const { sequelize, User, ChatHistory, StudyRecord, Question } = require('./models');

async function syncDatabase() {
  try {
    console.log('开始同步数据库...');

    // 强制同步所有模型
    await sequelize.sync({ force: false, alter: true });

    console.log('数据库同步成功！');

    // 检查表是否存在
    const tables = await sequelize.getQueryInterface().showAllTables();
    console.log('\n所有数据表:');
    tables.forEach(table => console.log(`  - ${table}`));

    // 检查 StudyRecord 表结构
    if (tables.includes('StudyRecords')) {
      const tableInfo = await sequelize.getQueryInterface().describeTable('StudyRecords');
      console.log('\nStudyRecords 表结构:');
      Object.keys(tableInfo).forEach(col => {
        console.log(`  ${col}: ${tableInfo[col].type}`);
      });
    }

    await sequelize.close();

  } catch (error) {
    console.error('同步失败:', error);
  }
}

syncDatabase();