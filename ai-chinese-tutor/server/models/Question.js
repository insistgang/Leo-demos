const { DataTypes } = require('sequelize');
const sequelize = require('../config/database');

const Question = sequelize.define('Question', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  type: {
    type: DataTypes.ENUM('pinyin', 'vocabulary', 'literature', 'idiom', 'correction', 'reading', 'grammar', 'comprehension'),
    allowNull: false,
    comment: '题型：拼音、词汇、文学、成语、纠错、阅读、语法、理解'
  },
  year: {
    type: DataTypes.INTEGER,
    allowNull: true,
    comment: '年份：2014-2024'
  },
  source: {
    type: DataTypes.STRING(100),
    allowNull: true,
    comment: '来源：如"2023年全国甲卷"'
  },
  content: {
    type: DataTypes.TEXT,
    allowNull: false,
    comment: '题干内容，支持 Markdown'
  },
  options: {
    type: DataTypes.TEXT,
    allowNull: true,
    comment: '选项，JSON 格式存储',
    get() {
      const value = this.getDataValue('options');
      return value ? JSON.parse(value) : null;
    },
    set(value) {
      this.setDataValue('options', JSON.stringify(value || {}));
    }
  },
  answer: {
    type: DataTypes.TEXT,
    allowNull: true,
    comment: '正确答案'
  },
  explanation: {
    type: DataTypes.TEXT,
    allowNull: true,
    comment: '解析'
  },
  difficulty: {
    type: DataTypes.INTEGER,
    allowNull: false,
    defaultValue: 3,
    comment: '难度等级：1-5'
  },
  questionNumber: {
    type: DataTypes.INTEGER,
    allowNull: true,
    comment: '题号'
  },
  tags: {
    type: DataTypes.TEXT,
    allowNull: true,
    comment: '标签，JSON 数组格式',
    get() {
      const value = this.getDataValue('tags');
      return value ? JSON.parse(value) : [];
    },
    set(value) {
      this.setDataValue('tags', JSON.stringify(value || []));
    }
  }
}, {
  tableName: 'Questions',
  indexes: [
    {
      fields: ['type']
    },
    {
      fields: ['year']
    },
    {
      fields: ['difficulty']
    },
    {
      fields: ['type', 'year']
    }
  ]
});

module.exports = Question;