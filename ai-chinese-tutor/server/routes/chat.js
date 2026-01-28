const express = require('express');
const { ChatHistory } = require('../models');
const authMiddleware = require('../middleware/auth');

const router = express.Router();

// 获取聊天历史
router.get('/history', authMiddleware, async (req, res) => {
  try {
    const userId = req.user.id;
    const { limit = 50, offset = 0 } = req.query;

    const chatHistories = await ChatHistory.findAll({
      where: { userId },
      order: [['createdAt', 'ASC']],
      limit: parseInt(limit),
      offset: parseInt(offset)
    });

    res.json({
      success: true,
      data: chatHistories
    });
  } catch (error) {
    console.error('获取聊天历史错误:', error);
    res.status(500).json({
      success: false,
      message: '获取聊天历史失败'
    });
  }
});

// 保存聊天消息
router.post('/message', authMiddleware, async (req, res) => {
  try {
    const userId = req.user.id;
    const { role, content } = req.body;

    // 验证必填字段
    if (!role || !content) {
      return res.status(400).json({
        success: false,
        message: '角色和内容不能为空'
      });
    }

    // 验证角色
    if (!['user', 'assistant'].includes(role)) {
      return res.status(400).json({
        success: false,
        message: '角色必须是 user 或 assistant'
      });
    }

    // 创建聊天记录
    const chatMessage = await ChatHistory.create({
      userId,
      role,
      content
    });

    res.status(201).json({
      success: true,
      message: '消息保存成功',
      data: chatMessage
    });
  } catch (error) {
    console.error('保存聊天消息错误:', error);
    res.status(500).json({
      success: false,
      message: '保存消息失败'
    });
  }
});

// 批量保存聊天消息（用于同步）
router.post('/messages', authMiddleware, async (req, res) => {
  try {
    const userId = req.user.id;
    const { messages } = req.body;

    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        success: false,
        message: '消息列表不能为空'
      });
    }

    // 验证并过滤消息
    const validMessages = messages.filter(msg => {
      return msg.role && msg.content && ['user', 'assistant'].includes(msg.role);
    });

    if (validMessages.length === 0) {
      return res.status(400).json({
        success: false,
        message: '没有有效的消息'
      });
    }

    // 批量创建
    const chatMessages = await ChatHistory.bulkCreate(
      validMessages.map(msg => ({
        userId,
        role: msg.role,
        content: msg.content
      }))
    );

    res.status(201).json({
      success: true,
      message: `成功保存 ${chatMessages.length} 条消息`,
      data: chatMessages
    });
  } catch (error) {
    console.error('批量保存聊天消息错误:', error);
    res.status(500).json({
      success: false,
      message: '批量保存失败'
    });
  }
});

// 清空聊天记录
router.delete('/clear', authMiddleware, async (req, res) => {
  try {
    const userId = req.user.id;

    const deletedCount = await ChatHistory.destroy({
      where: { userId }
    });

    res.json({
      success: true,
      message: `已清空 ${deletedCount} 条聊天记录`
    });
  } catch (error) {
    console.error('清空聊天记录错误:', error);
    res.status(500).json({
      success: false,
      message: '清空失败'
    });
  }
});

// 删除特定时间之前的聊天记录
router.delete('/history/before/:date', authMiddleware, async (req, res) => {
  try {
    const userId = req.user.id;
    const { date } = req.params;

    const targetDate = new Date(date);
    if (isNaN(targetDate.getTime())) {
      return res.status(400).json({
        success: false,
        message: '无效的日期格式'
      });
    }

    const deletedCount = await ChatHistory.destroy({
      where: {
        userId,
        createdAt: {
          [require('sequelize').Op.lt]: targetDate
        }
      }
    });

    res.json({
      success: true,
      message: `已删除 ${deletedCount} 条历史记录`
    });
  } catch (error) {
    console.error('删除聊天记录错误:', error);
    res.status(500).json({
      success: false,
      message: '删除失败'
    });
  }
});

module.exports = router;