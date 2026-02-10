const sequelize = require('../config/database');
const User = require('./User');
const ChatHistory = require('./ChatHistory');
const StudyRecord = require('./StudyRecord');
const Question = require('./Question');

// 定义模型之间的关联
User.hasMany(ChatHistory, { foreignKey: 'userId', as: 'chatHistories' });
ChatHistory.belongsTo(User, { foreignKey: 'userId', as: 'user' });

User.hasMany(StudyRecord, { foreignKey: 'userId', as: 'studyRecords' });
StudyRecord.belongsTo(User, { foreignKey: 'userId', as: 'user' });

module.exports = {
  sequelize,
  User,
  ChatHistory,
  StudyRecord,
  Question
};