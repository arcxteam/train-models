# Gunicorn config variables
loglevel = "info"
errorlog = "-"  # stderr
accesslog = "-"  # stdout
worker_tmp_dir = "/dev/shm"
graceful_timeout = 120
timeout = 30
keepalive = 5
worker_class = "gthread"
workers = 2
threads = 6
bind = "0.0.0.0:8001"
bind = "0.0.0.0:8002"
bind = "0.0.0.0:8003"
