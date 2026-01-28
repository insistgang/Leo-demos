const express = require('express');
const { Question, sequelize } = require('../models');
const authMiddleware = require('../middleware/auth');
const { Op } = require('sequelize');

const router = express.Router();

// 静态路由（不包含参数）要放在动态路由之前

// 获取题目统计信息
router.get('/stats', authMiddleware, async (req, res) => {
  try {
    const stats = await Question.findAll({
      attributes: [
        'type',
        [sequelize.fn('COUNT', '*'), 'count']
      ],
      group: ['type'],
      raw: true
    });

    // 转换为对象格式
    const statsObj = {};
    stats.forEach(s => {
      statsObj[s.type] = parseInt(s.count);
    });

    res.json({
      success: true,
      data: statsObj
    });
  } catch (error) {
    console.error('获取统计失败:', error);
    res.status(500).json({
      success: false,
      message: '获取统计失败'
    });
  }
});

// 获取随机题目
router.get('/random', authMiddleware, async (req, res) => {
  try {
    const { type, count = 10, difficulty, year } = req.query;

    const whereCondition = {};
    if (type) whereCondition.type = type;
    if (difficulty) whereCondition.difficulty = difficulty;
    if (year) whereCondition.year = parseInt(year);

    const questions = await Question.findAll({
      where: whereCondition,
      order: sequelize.literal('RANDOM()'),
      limit: parseInt(count)
    });

    res.json({
      success: true,
      data: questions
    });
  } catch (error) {
    console.error('获取题目失败:', error);
    res.status(500).json({
      success: false,
      message: '获取题目失败'
    });
  }
});

// 搜索题目
router.get('/search', authMiddleware, async (req, res) => {
  try {
    const { keyword, type, limit = 20 } = req.query;

    const whereCondition = {};
    if (keyword) {
      whereCondition[Op.or] = [
        { content: { [Op.like]: `%${keyword}%` } },
        { explanation: { [Op.like]: `%${keyword}%` } }
      ];
    }
    if (type) whereCondition.type = type;

    const questions = await Question.findAll({
      where: whereCondition,
      limit: parseInt(limit)
    });

    res.json({
      success: true,
      data: questions
    });
  } catch (error) {
    console.error('搜索题目失败:', error);
    res.status(500).json({
      success: false,
      message: '搜索题目失败'
    });
  }
});

// 动态路由（包含参数）要放在最后

// 根据类型获取题目
router.get('/type/:type', authMiddleware, async (req, res) => {
  try {
    const { type } = req.params;
    const { count = 10, difficulty, year } = req.query;

    const whereCondition = { type };
    if (difficulty) whereCondition.difficulty = difficulty;
    if (year) whereCondition.year = parseInt(year);

    const questions = await Question.findAll({
      where: whereCondition,
      order: sequelize.literal('RANDOM()'),
      limit: parseInt(count)
    });

    res.json({
      success: true,
      data: questions
    });
  } catch (error) {
    console.error('获取题目失败:', error);
    res.status(500).json({
      success: false,
      message: '获取题目失败'
    });
  }
});

// 获取指定年份的题目
router.get('/year/:year', authMiddleware, async (req, res) => {
  try {
    const { year } = req.params;
    const { type, count = 20, difficulty } = req.query;

    const whereCondition = { year: parseInt(year) };
    if (type) whereCondition.type = type;
    if (difficulty) whereCondition.difficulty = difficulty;

    const questions = await Question.findAll({
      where: whereCondition,
      order: [['questionNumber', 'ASC']],
      limit: parseInt(count)
    });

    res.json({
      success: true,
      data: questions
    });
  } catch (error) {
    console.error('获取题目失败:', error);
    res.status(500).json({
      success: false,
      message: '获取题目失败'
    });
  }
});

// 获取单个题目详情（必须放在最后）
router.get('/:id', authMiddleware, async (req, res) => {
  try {
    const { id } = req.params;

    // 如果传入的是数字，按主键查找
    if (!isNaN(id)) {
      const question = await Question.findByPk(parseInt(id));

      if (!question) {
        return res.status(404).json({
          success: false,
          message: '题目不存在'
        });
      }

      res.json({
        success: true,
        data: question
      });
    } else {
      // 如果不是数字，返回404
      return res.status(404).json({
        success: false,
        message: '无效的题目ID'
      });
    }
  } catch (error) {
    console.error('获取题目详情失败:', error);
    res.status(500).json({
      success: false,
      message: '获取题目详情失败'
    });
  }
});

module.exports = router;