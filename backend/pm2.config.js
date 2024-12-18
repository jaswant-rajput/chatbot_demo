module.exports = {
  apps : [
    {
    name   : "flask",
    // -w means workers (2 x CPU Cores) + 1, see how many threads your prod server has in each Core
    script: "./venv/bin/gunicorn -w 15 --threads 2 -b 0.0.0.0:8000 app:app",
    max_restarts:10,
  },
  {
    name   : "redis-logging-queue",
    script: "./venv/bin/rq worker logging_queue",
    max_restarts:10,
  },
]
}
