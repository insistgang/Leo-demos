const bcrypt = require('bcryptjs');
const { User } = require('./models');

async function createDefaultUser() {
  try {
    console.log('创建默认用户...');

    // 检查是否已存在 admin 用户
    const existingAdmin = await User.findOne({ where: { username: 'admin' } });
    if (existingAdmin) {
      console.log('admin 用户已存在，跳过创建');
    } else {
      // 创建 admin 用户
      const hashedPassword = await bcrypt.hash('123456', 10);

      const admin = await User.create({
        username: 'admin',
        password: hashedPassword,
        nickname: '管理员',
        email: 'admin@example.com'
      });

      console.log('✅ 成功创建管理员用户');
      console.log('   用户名: admin');
      console.log('   密码: 123456');
      console.log('   昵称: 管理员');
    }

    // 检查是否已存在 test 用户
    const existingTest = await User.findOne({ where: { username: 'test' } });
    if (existingTest) {
      console.log('\ntest 用户已存在，跳过创建');
    } else {
      // 创建 test 用户
      const hashedPassword = await bcrypt.hash('123456', 10);

      await User.create({
        username: 'test',
        password: hashedPassword,
        nickname: '测试用户',
        email: 'test@example.com'
      });

      console.log('\n✅ 成功创建测试用户');
      console.log('   用户名: test');
      console.log('   密码: 123456');
      console.log('   昵称: 测试用户');
    }

    console.log('\n所有现有用户：');
    const allUsers = await User.findAll({
      attributes: ['id', 'username', 'nickname', 'email'],
      raw: true
    });

    allUsers.forEach(user => {
      console.log(`- ${user.username} (ID: ${user.id})`);
    });

  } catch (error) {
    console.error('创建用户失败:', error);
  }
}

createDefaultUser();