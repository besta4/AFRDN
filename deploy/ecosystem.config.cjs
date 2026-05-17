module.exports = {
  apps: [
    {
      name: "jatayu-backend",
      cwd: "/var/www/jatayu/backend",
      script: ".venv/bin/python",
      args: "-m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 2",
      interpreter: "none",
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
  ],
};
