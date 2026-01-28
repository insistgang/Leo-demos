# 部署指南

本文档指导如何将 AI 语文助手项目部署到生产服务器。

## 前提条件

- 服务器安装了 Node.js (版本 16 或以上)
- 服务器安装了 npm 或 yarn
- 服务器安装了 Nginx (可选，用于反向代理)
- 服务器安装了 PM2 (进程管理器)

## 部署步骤

### 1. 准备服务器

```bash
# 安装 Node.js
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# 安装 PM2
sudo npm install pm2@latest -g

# 创建项目目录
sudo mkdir -p /var/www/ai-tutor
sudo chown -R $USER:$USER /var/www/ai-tutor
cd /var/www/ai-tutor
```

### 2. 上传代码

将以下文件和文件夹上传到服务器：

```
ai-tutor/
├── dist/                    # 前端构建后的文件
├── server/                  # 后端代码
│   ├── src/
│   ├── models/
│   ├── routes/
│   ├── package.json
│   ├── ecosystem.config.js  # PM2 配置
│   └── .env.production      # 生产环境配置
├── package.json             # 前端依赖
└── nginx.conf              # Nginx 配置示例
```

### 3. 配置环境变量

编辑 `server/.env.production` 文件：

```env
NODE_ENV=production
PORT=3001
JWT_SECRET=your-strong-jwt-secret-here
DEEPSEEK_API_KEY=your-deepseek-api-key
DATABASE_PATH=./database/production.sqlite
```

### 4. 安装依赖并构建

```bash
# 进入前端目录
cd /var/www/ai-tutor

# 安装前端依赖（如果还没构建）
npm install

# 构建前端（如果在本地已构建可跳过）
npm run build

# 进入后端目录
cd server

# 安装后端依赖
npm install --production
```

### 5. 初始化数据库

```bash
cd /var/www/ai-tutor/server

# 同步数据库
node -e "require('./models').sequelize.sync()"
```

### 6. 启动后端服务

使用 PM2 启动后端服务：

```bash
# 创建日志目录
mkdir -p logs

# 使用 PM2 启动
pm2 start ecosystem.config.js

# 查看服务状态
pm2 status

# 设置开机自启
pm2 startup
pm2 save
```

### 7. 配置 Nginx

将 `nginx.conf` 配置添加到 Nginx：

```bash
# 复制配置文件
sudo cp /var/www/ai-tutor/nginx.conf /etc/nginx/sites-available/ai-tutor

# 创建软链接
sudo ln -s /etc/nginx/sites-available/ai-tutor /etc/nginx/sites-enabled/

# 修改配置中的路径
sudo nano /etc/nginx/sites-available/ai-tutor

# 测试配置
sudo nginx -t

# 重启 Nginx
sudo systemctl restart nginx
```

### 8. 防火墙设置

```bash
# 如果使用 UFW
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 3001/tcp  # 如果需要直接访问后端
```

## 维护命令

### PM2 常用命令

```bash
# 查看服务状态
pm2 status

# 查看日志
pm2 logs ai-tutor-server

# 重启服务
pm2 restart ai-tutor-server

# 停止服务
pm2 stop ai-tutor-server

# 查看监控
pm2 monit

# 更新代码后重启
pm2 reload ai-tutor-server
```

### 备份数据库

```bash
# 备份 SQLite 数据库
cp /var/www/ai-tutor/server/database/production.sqlite /backup/ai-tutor-$(date +%Y%m%d).sqlite
```

### 更新部署

```bash
# 1. 上传新代码
# 2. 进入前端目录构建
cd /var/www/ai-tutor
npm run build

# 3. 进入后端目录
cd server
npm install --production  # 如果有新依赖

# 4. 重启服务
pm2 restart ai-tutor-server
```

## 注意事项

1. **安全性**：
   - 使用强密码作为 JWT_SECRET
   - 不要将 .env 文件提交到版本控制
   - 定期更新依赖包

2. **性能优化**：
   - 启用 Nginx 的 Gzip 压缩
   - 配置适当的缓存策略
   - 监控服务器资源使用情况

3. **监控**：
   - 使用 PM2 监控应用状态
   - 配置日志轮转
   - 设置告警通知

## 故障排查

### 查看错误日志

```bash
# PM2 日志
pm2 logs ai-tutor-server --err

# Nginx 日志
sudo tail -f /var/log/nginx/error.log
```

### 常见问题

1. **端口被占用**：
   - 检查端口使用情况：`sudo lsof -i :3001`
   - 修改 .env.production 中的 PORT

2. **数据库连接失败**：
   - 检查数据库文件路径
   - 确保有写入权限

3. **API 请求失败**：
   - 检查 Nginx 配置
   - 确认后端服务正在运行
   - 检查防火墙设置

## SSL/HTTPS 配置

建议使用 Let's Encrypt 免费证书：

```bash
# 安装 Certbot
sudo apt install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d your-domain.com

# 自动续期
sudo crontab -e
# 添加：0 12 * * * /usr/bin/certbot renew --quiet
```

## 域名解析

将域名解析到服务器 IP：

```
A     @        你的服务器IP
A     www      你的服务器IP
```

部署完成后，通过浏览器访问你的域名即可使用应用。