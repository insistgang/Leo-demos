const { DataTypes } = require('sequelize');
const sequelize = require('../config/database');
const User = require('./User');

const StudyRecord = sequelize.define('StudyRecord', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  userId: {
    type: DataTypes.INTEGER,
    allowNull: false,
    references: {
      model: User,
      key: 'id'
    },
    field: 'user_id'
  },
  type: {
    type: DataTypes.STRING(50),
    allowNull: false,
    comment: '学习类型：vocabulary, literature, idiom, exercise, pinyin, correction'
  },
  content: {
    type: DataTypes.TEXT,
    allowNull: false,
    comment: '学习内容，JSON 格式存储',
    get() {
      const value = this.getDataValue('content');
      return value ? JSON.parse(value) : null;
    },
    set(value) {
      this.setDataValue('content', JSON.stringify(value));
    }
  },
  duration: {
    type: DataTypes.INTEGER,
    allowNull: false,
    defaultValue: 0,
    comment: '学习时长（秒）'
  },
  score: {
    type: DataTypes.INTEGER,
    allowNull: true,
    comment: '得分（可选）'
  },
  totalQuestions: {
    type: DataTypes.INTEGER,
    allowNull: true,
    comment: '总题数（练习类型）'
  },
  correctAnswers: {
    type: DataTypes.INTEGER,
    allowNull: true,
    comment: '正确答案数（练习类型）'
  }
}, {
  tableName: 'StudyRecord',
  indexes: [
    {
      fields: ['user_id', 'type', 'createdAt']
    },
    {
      fields: ['user_id', 'createdAt']
    }
  ]
});

// 关联在 models/index.js 中定义
module.exports = StudyRecord;