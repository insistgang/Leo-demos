const express = require('express');
const { StudyRecord } = require('../models');
const authMiddleware = require('../middleware/auth');

const router = express.Router();

// 添加学习记录（带调试日志）
router.post('/record', authMiddleware, async (req, res) => {
  try {
    console.log('\n=== 收到保存学习记录请求 ===');
    console.log('用户ID:', req.user?.id);
    console.log('请求体:', JSON.stringify(req.body, null, 2));

    const userId = req.user.id;
    const { type, content, duration, score, totalQuestions, correctAnswers } = req.body;

    // 验证必填字段
    if (!type) {
      console.log('错误：缺少 type 字段');
      return res.status(400).json({
        success: false,
        message: '学习类型不能为空'
      });
    }

    if (!content) {
      console.log('错误：缺少 content 字段');
      return res.status(400).json({
        success: false,
        message: '学习内容不能为空'
      });
    }

    // 创建学习记录
    const studyRecord = await StudyRecord.create({
      userId,
      type,
      content,
      duration: duration || 0,
      score: score || 0,
      totalQuestions: totalQuestions || 0,
      correctAnswers: correctAnswers || 0
    });

    console.log('记录创建成功，ID:', studyRecord.id);
    console.log('========================\n');

    res.status(201).json({
      success: true,
      message: '学习记录添加成功',
      data: studyRecord
    });
  } catch (error) {
    console.error('添加学习记录错误:', error);
    res.status(500).json({
      success: false,
      message: '添加学习记录失败',
      error: error.message
    });
  }
});

// 获取学习记录（带调试日志）
router.get('/records', authMiddleware, async (req, res) => {
  try {
    console.log('\n=== 查询学习记录 ===');
    const userId = req.user.id;
    console.log('用户ID:', userId);

    const studyRecords = await StudyRecord.findAll({
      where: { userId },
      order: [['createdAt', 'DESC']],
      limit: 10
    });

    console.log('找到记录数:', studyRecords.length);
    studyRecords.forEach(r => {
      console.log(`- ID: ${r.id}, 类型: ${r.type}, 得分: ${r.score}, 时间: ${r.createdAt}`);
    });
    console.log('==================\n');

    res.json({
      success: true,
      data: studyRecords
    });
  } catch (error) {
    console.error('获取学习记录错误:', error);
    res.status(500).json({
      success: false,
      message: '获取学习记录失败'
    });
  }
});

module.exports = router;