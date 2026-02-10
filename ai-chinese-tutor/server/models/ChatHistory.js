const { DataTypes } = require('sequelize');
const sequelize = require('../config/database');
const User = require('./User');

const ChatHistory = sequelize.define('ChatHistory', {
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
  role: {
    type: DataTypes.ENUM('user', 'assistant'),
    allowNull: false
  },
  content: {
    type: DataTypes.TEXT,
    allowNull: false,
    comment: '支持 Markdown 格式的消息内容'
  }
}, {
  tableName: 'ChatHistory',
  indexes: [
    {
      fields: ['user_id', 'createdAt']
    }
  ]
});

// 关联在 models/index.js 中定义
module.exports = ChatHistory;