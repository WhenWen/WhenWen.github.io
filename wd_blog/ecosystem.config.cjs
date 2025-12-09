module.exports = {
  apps: [{
    name: 'wd-blog',
    script: 'serve.js',
    cwd: '/root/HiddenHomepage/wd_blog',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '200M',
    env: {
      NODE_ENV: 'production',
      PORT: 7292
    }
  }]
};
