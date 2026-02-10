const express = require('express');
const { StudyRecord } = require('../models');
const { Op } = require('sequelize');
const authMiddleware = require('../middleware/auth');

const router = express.Router();

// 获取学习记录
router.get('/records', authMiddleware, async (req, res) => {
  try {
    const userId = req.user.id;
    const { type, limit = 50, offset = 0, startDate, endDate } = req.query;

    // 构建查询条件
    const whereCondition = { userId };
    if (type) {
      whereCondition.type = type;
    }
    if (startDate || endDate) {
      whereCondition.createdAt = {};
      if (startDate) {
        whereCondition.createdAt[Op.gte] = new Date(startDate);
      }
      if (endDate) {
        whereCondition.createdAt[Op.lte] = new Date(endDate + ' 23:59:59');
      }
    }

    const studyRecords = await StudyRecord.findAll({
      where: whereCondition,
      order: [['createdAt', 'DESC']],
      limit: parseInt(limit),
      offset: parseInt(offset)
    });

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

// 添加学习记录
router.post('/record', authMiddleware, async (req, res) => {
  try {
    const userId = req.user.id;
    const { type, content, duration, score, totalQuestions, correctAnswers } = req.body;

    // 验证必填字段
    if (!type || !content) {
      return res.status(400).json({
        success: false,
        message: '学习类型和内容不能为空'
      });
    }

    // 验证学习类型
    const validTypes = ['vocabulary', 'literature', 'idiom', 'exercise', 'pinyin', 'correction'];
    if (!validTypes.includes(type)) {
      return res.status(400).json({
        success: false,
        message: '无效的学习类型'
      });
    }

    // 创建学习记录
    const studyRecord = await StudyRecord.create({
      userId,
      type,
      content,
      duration: duration || 0,
      score,
      totalQuestions,
      correctAnswers
    });

    res.status(201).json({
      success: true,
      message: '学习记录添加成功',
      data: studyRecord
    });
  } catch (error) {
    console.error('添加学习记录错误:', error);
    res.status(500).json({
      success: false,
      message: '添加学习记录失败'
    });
  }
});

// 批量添加学习记录
router.post('/records', authMiddleware, async (req, res) => {
  try {
    const userId = req.user.id;
    const { records } = req.body;

    if (!Array.isArray(records) || records.length === 0) {
      return res.status(400).json({
        success: false,
        message: '记录列表不能为空'
      });
    }

    // 验证并过滤记录
    const validTypes = ['vocabulary', 'literature', 'idiom', 'exercise', 'pinyin', 'correction'];
    const validRecords = records.filter(record => {
      return record.type && record.content && validTypes.includes(record.type);
    });

    if (validRecords.length === 0) {
      return res.status(400).json({
        success: false,
        message: '没有有效的学习记录'
      });
    }

    // 批量创建
    const studyRecords = await StudyRecord.bulkCreate(
      validRecords.map(record => ({
        userId,
        type: record.type,
        content: record.content,
        duration: record.duration || 0,
        score: record.score,
        totalQuestions: record.totalQuestions,
        correctAnswers: record.correctAnswers
      }))
    );

    res.status(201).json({
      success: true,
      message: `成功添加 ${studyRecords.length} 条学习记录`,
      data: studyRecords
    });
  } catch (error) {
    console.error('批量添加学习记录错误:', error);
    res.status(500).json({
      success: false,
      message: '批量添加失败'
    });
  }
});

// 获取学习统计
router.get('/statistics', authMiddleware, async (req, res) => {
  try {
    const userId = req.user.id;
    const { days = 30 } = req.query;

    // 计算起始日期
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - parseInt(days));

    // 获取总学习时长
    const totalDuration = await StudyRecord.sum('duration', {
      where: {
        userId,
        createdAt: {
          [Op.gte]: startDate
        }
      }
    }) || 0;

    // 获取学习天数
    const studyDays = await StudyRecord.count({
      distinct: true,
      col: 'createdAt',
      where: {
        userId,
        createdAt: {
          [Op.gte]: startDate
        }
      }
    });

    // 获取各类型的学习记录数量
    const typeStats = await StudyRecord.findAll({
      attributes: [
        'type',
        [require('sequelize').fn('COUNT', '*'), 'count'],
        [require('sequelize').fn('SUM', require('sequelize').col('duration')), 'totalDuration']
      ],
      where: {
        userId,
        createdAt: {
          [Op.gte]: startDate
        }
      },
      group: ['type']
    });

    // 获取练习统计
    const exerciseStats = await StudyRecord.findAll({
      attributes: [
        [require('sequelize').fn('AVG', require('sequelize').col('score')), 'averageScore'],
        [require('sequelize').fn('COUNT', '*'), 'totalExercises'],
        [require('sequelize').fn('SUM', require('sequelize').col('totalQuestions')), 'totalQuestions'],
        [require('sequelize').fn('SUM', require('sequelize').col('correctAnswers')), 'totalCorrect']
      ],
      where: {
        userId,
        type: 'exercise',
        score: {
          [Op.not]: null
        },
        createdAt: {
          [Op.gte]: startDate
        }
      }
    });

    // 格式化统计数据
    const statistics = {
      totalDuration,
      studyDays,
      averageDailyDuration: studyDays > 0 ? Math.round(totalDuration / studyDays) : 0,
      typeStatistics: typeStats.map(stat => ({
        type: stat.type,
        count: parseInt(stat.dataValues.count),
        totalDuration: parseInt(stat.dataValues.totalDuration) || 0
      })),
      exerciseStatistics: exerciseStats.length > 0 ? {
        averageScore: Math.round(exerciseStats[0].dataValues.averageScore * 100) / 100,
        totalExercises: parseInt(exerciseStats[0].dataValues.totalExercises),
        totalQuestions: parseInt(exerciseStats[0].dataValues.totalQuestions) || 0,
        totalCorrect: parseInt(exerciseStats[0].dataValues.totalCorrect) || 0,
        accuracy: exerciseStats[0].dataValues.totalQuestions > 0
          ? Math.round((exerciseStats[0].dataValues.totalCorrect / exerciseStats[0].dataValues.totalQuestions) * 100 * 100) / 100
          : 0
      } : null
    };

    res.json({
      success: true,
      data: statistics
    });
  } catch (error) {
    console.error('获取学习统计错误:', error);
    res.status(500).json({
      success: false,
      message: '获取学习统计失败'
    });
  }
});

// 删除学习记录
router.delete('/record/:id', authMiddleware, async (req, res) => {
  try {
    const userId = req.user.id;
    const { id } = req.params;

    const deletedCount = await StudyRecord.destroy({
      where: {
        id,
        userId
      }
    });

    if (deletedCount === 0) {
      return res.status(404).json({
        success: false,
        message: '学习记录不存在'
      });
    }

    res.json({
      success: true,
      message: '学习记录删除成功'
    });
  } catch (error) {
    console.error('删除学习记录错误:', error);
    res.status(500).json({
      success: false,
      message: '删除失败'
    });
  }
});

module.exports = router;