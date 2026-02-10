const { User } = require('./models');

async function checkUsers() {
  try {
    const users = await User.findAll({
      attributes: ['id', 'username', 'nickname', 'email'],
      raw: true
    });

    console.log('系统中的用户：');
    if (users.length === 0) {
      console.log('\n没有找到任何用户！');
      console.log('\n请按以下步骤创建用户：');
      console.log('1. 打开浏览器访问：http://localhost:3000');
      console.log('2. 点击登录页面的"注册"按钮');
      console.log('3. 填写用户名、密码等信息完成注册');
      console.log('4. 使用注册的账号登录系统');
    } else {
      console.log('\n找到以下用户：');
      users.forEach(user => {
        console.log(`- 用户名: ${user.username}`);
        console.log(`  昵称: ${user.nickname || '未设置'}`);
        console.log(`  邮箱: ${user.email || '未设置'}`);
        console.log(`  ID: ${user.id}\n`);
      });
    }
  } catch (error) {
    console.error('查询用户失败:', error);
  }
}

checkUsers();