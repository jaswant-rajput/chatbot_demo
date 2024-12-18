# BeyondChats Flask App

## Documentation
[Notion Docs](
https://beyondexams.notion.site/Setup-Flask-server-da57385af603482ca37fd752250ebbd5)

### Dev Setup
1. Install and Setup latest version of [Python](https://www.python.org/downloads/)
2. Create Virtual Environment
```bash
python3 -m venv venv
```
3. Active Virtual Environment
Linux/Mac
```bash
source ./venv/bin/activate
```
Windows
```cmd
venv\Scripts\activate
```
4. Install Dependencies
```bash
pip install -r requirements.txt
```
5. Create a new settings.py file from settings.example.py and update it accordingly
```bash
cp settings.example.py settings.py
```
6. Create a new credentials.py file from constants/credentials.sample.py and update it accordingly
```bash
cp constants/credentials.sample.py constants/credentials.py
```
7. Install punkt tokenizer
8. Run app
```bash
python app.py
```

# install punkt tokenizer
```python
import nltk
nltk.download("punkt")
```


# Prod Server Setup

### install NVM
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
nvm install --lts
```

### install gunicorn in venv
```bash
pip install gunicorn
```

### Install PM2
```bash
npm install pm2 -g
pm2 install pm2-logrotate
```

### Install redis-server
Install Redis and start the server (defult port 6379, if you change the port update the `settings.py` file)
```bash
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list

sudo apt-get update
sudo apt-get install redis
```

```python
REDIS_HOST = "localhost"
REDIS_PORT = 6379
```
Add Redis server to services
```bash
 sudo systemctl enable redis-server
 sudo systemctl start redis-server
```

### start prod server
```bash
pm2 start pm2.config.js
pm2 save
```

### Create PM2 service for auto start - run this then follow instruction
```bash
pm2 startup
```

### PM2 Logs by default are stored in 
```bash
ls /.pm2/logs
```

### OLD README
<!-- # GPT3 -->

### Steps

1. Install the OpenAI module with `pip install openai`
2. Run `python --version` to verify the system version of python
3. Clone down this repo with `git clone git@gitlab.com:bishalbar77/be_gpt3.git`
4. If you want to run the GPT-3 on custom knowledge base then and copy your data into `data.txt` in `<<DATA>>` block
5. Go into the repo with `cd be_gpt3`
6. Run the demo script with `python chat.py`