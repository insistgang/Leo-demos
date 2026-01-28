const { Question, sequelize } = require('./models');

async function updateQuestionsWithAnswers() {
  try {
    console.log('更新题目答案...');

    // 获取所有题目
    const questions = await Question.findAll();
    console.log(`找到 ${questions.length} 道题目`);

    // 加载答案
    const fs = require('fs');
    const path = require('path');
    const answersFile = path.join(__dirname, '../k12_md/answers_2015_2024.txt');
    const answers = new Map();

    const content = fs.readFileSync(answersFile, 'utf-8');
    const lines = content.split('\n');

    lines.forEach(line => {
      const match = line.match(/^(\d{4})-(\d+):([A-D])$/);
      if (match) {
        const key = `${match[1]}-${match[2]}`;
        answers.set(key, match[3]);
      }
    });

    console.log(`已加载 ${answers.size} 个答案`);

    // 更新每道题目的答案
    let updatedCount = 0;
    for (const q of questions) {
      const answerKey = `${q.year}-${q.questionNumber}`;
      const correctAnswer = answers.get(answerKey);

      if (correctAnswer && q.answer !== correctAnswer) {
        await q.update({ answer: correctAnswer });
        updatedCount++;
        console.log(`更新题目 ${answerKey}: ${q.answer} -> ${correctAnswer}`);
      }
    }

    console.log(`\n成功更新 ${updatedCount} 道题目的答案`);

    // 验证更新结果
    const sampleQuestions = await Question.findAll({
      where: { type: 'pinyin' },
      limit: 5
    });

    console.log('\n验证更新后的拼音题目：');
    sampleQuestions.forEach((q, index) => {
      console.log(`\n题目${index + 1} (${q.year}-${q.questionNumber}):`);
      console.log('答案:', q.answer);
    });

    await sequelize.close();

  } catch (error) {
    console.error('更新失败:', error);
  }
}

updateQuestionsWithAnswers();