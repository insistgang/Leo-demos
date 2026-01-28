const { sequelize, Question } = require('./models');
const fs = require('fs');

async function exportQuestions() {
  try {
    console.log('=== 导出题目数据 ===\n');

    // 连接数据库
    await sequelize.authenticate();
    console.log('数据库连接成功');

    // 查询所有题目
    const questions = await Question.findAll({
      raw: true
    });

    console.log(`找到 ${questions.length} 道题目`);

    // 按类型统计
    const typeStats = {};
    questions.forEach(q => {
      if (!typeStats[q.type]) {
        typeStats[q.type] = 0;
      }
      typeStats[q.type]++;
    });

    console.log('\n题目类型统计:');
    Object.entries(typeStats).forEach(([type, count]) => {
      console.log(`  ${type}: ${count} 道`);
    });

    // 导出到 JSON 文件
    const exportData = JSON.stringify(questions, null, 2);
    fs.writeFileSync('../questions-export.json', exportData, 'utf8');
    console.log('\n✓ 题目已导出到 questions-export.json');

    // 生成 SQL 插入语句
    let sqlStatements = 'BEGIN TRANSACTION;\n';
    questions.forEach(q => {
      const content = JSON.stringify(q.content).replace(/'/g, "''");
      const options = JSON.stringify(q.options).replace(/'/g, "''");

      sqlStatements += `INSERT INTO Questions (type, year, source, content, options, answer, explanation, difficulty, questionNumber, tags, createdAt, updatedAt) VALUES ('${q.type}', ${q.year || 'NULL'}, '${q.source || ''}', '${content}', '${options}', '${q.answer}', '${q.explanation || ''}', '${q.difficulty || ''}', ${q.questionNumber || 'NULL'}, '${q.tags || ''}', '${q.createdAt}', '${q.updatedAt}');\n`;
    });
    sqlStatements += 'COMMIT;';

    fs.writeFileSync('../questions-insert.sql', sqlStatements, 'utf8');
    console.log('✓ SQL 插入语句已生成到 questions-insert.sql');

  } catch (error) {
    console.error('导出失败:', error);
  } finally {
    await sequelize.close();
  }
}

exportQuestions();