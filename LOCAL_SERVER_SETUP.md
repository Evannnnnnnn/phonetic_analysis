# Running Your Mac as a Server

## 🚀 Quick Start with ngrok (Recommended)

### 1. Install ngrok
```bash
brew install ngrok
```

### 2. Create a startup script
Create `start_server.sh`:
```bash
#!/bin/bash
cd /Users/evanlou/workspace/phonetic_analysis
export GEMINI_API_KEY="your-api-key-here"
uvicorn app:app --host 0.0.0.0 --port 8000
```

Make it executable:
```bash
chmod +x start_server.sh
```

### 3. Run your server
```bash
./start_server.sh
```

### 4. In another terminal, start ngrok
```bash
ngrok http 8000
```

### 5. Share the URL!
ngrok will display a URL like: `https://abc123.ngrok-free.app`
Share this URL with anyone - they can access your app!

---

## 🔄 Auto-Start on Mac Boot (Optional)

### Create a LaunchAgent

1. **Create the plist file:**
```bash
nano ~/Library/LaunchAgents/com.phonetic.app.plist
```

2. **Add this content:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.phonetic.app</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/evanlou/workspace/phonetic_analysis/start_server.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/evanlou/workspace/phonetic_analysis/logs/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/evanlou/workspace/phonetic_analysis/logs/stderr.log</string>
</dict>
</plist>
```

3. **Create logs directory:**
```bash
mkdir -p /Users/evanlou/workspace/phonetic_analysis/logs
```

4. **Load the service:**
```bash
launchctl load ~/Library/LaunchAgents/com.phonetic.app.plist
```

Now your app starts automatically when you boot your Mac!

---

## 🌐 Option: Port Forwarding with Dynamic DNS

### 1. Find your Mac's local IP
```bash
ipconfig getifaddr en0  # For WiFi
# or
ipconfig getifaddr en1  # For Ethernet
```

### 2. Set up port forwarding on router
- Login to your router (usually http://192.168.1.1)
- Find "Port Forwarding" or "Virtual Server"
- Forward port 8000 to your Mac's IP

### 3. Get your public IP
```bash
curl ifconfig.me
```

### 4. Set up Dynamic DNS
- Sign up at https://www.noip.com (free)
- Create a hostname (e.g., `myphonetic.ddns.net`)
- Install their DUC (Dynamic Update Client)
- Or use DuckDNS: https://www.duckdns.org

### 5. Run your server
```bash
./start_server.sh
```

### 6. Access via your domain
`http://myphonetic.ddns.net:8000`

---

## 🔒 Security Considerations

### Basic Security Setup

1. **Add rate limiting** to prevent abuse:
```bash
pip install slowapi
```

Add to `app.py`:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/align")
@limiter.limit("10/minute")  # 10 requests per minute
async def align_audio(...):
    ...
```

2. **Set up firewall:**
```bash
# Enable macOS firewall
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on

# Allow Python
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/bin/python3
```

3. **Use HTTPS with Let's Encrypt** (if using port forwarding):
```bash
brew install certbot
sudo certbot certonly --standalone -d yourdomain.ddns.net
```

---

## 📊 Monitoring Your Server

### View logs in real-time:
```bash
tail -f /Users/evanlou/workspace/phonetic_analysis/logs/stdout.log
```

### Check if server is running:
```bash
lsof -i :8000
```

### Stop the server:
```bash
# If running in terminal: Ctrl+C

# If running as LaunchAgent:
launchctl unload ~/Library/LaunchAgents/com.phonetic.app.plist
```

### Restart the server:
```bash
launchctl unload ~/Library/LaunchAgents/com.phonetic.app.plist
launchctl load ~/Library/LaunchAgents/com.phonetic.app.plist
```

---

## ⚡ Performance Tips

1. **Keep your Mac from sleeping:**
   - System Preferences → Energy Saver → Prevent computer from sleeping

2. **Close unnecessary apps** to save resources

3. **Monitor resource usage:**
```bash
top -o cpu  # CPU usage
top -o mem  # Memory usage
```

---

## 🆘 Troubleshooting

### Port already in use:
```bash
# Find what's using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>
```

### Can't access from outside network:
1. Check router port forwarding is correct
2. Check macOS firewall allows connections
3. Verify your public IP: `curl ifconfig.me`
4. Test locally first: `http://localhost:8000`

### ngrok "too many connections":
- Free tier has connection limits
- Upgrade to paid plan or use alternatives like:
  - localtunnel: `npm install -g localtunnel`
  - serveo: `ssh -R 80:localhost:8000 serveo.net`

---

## 💰 Cost Comparison

| Method | Cost | Effort | Reliability |
|--------|------|--------|-------------|
| ngrok (free) | Free | Very Easy | Good* |
| ngrok (paid) | $8/month | Very Easy | Excellent |
| Port Forward + DDNS | Free | Medium | Good* |
| Tailscale | Free | Easy | Excellent |
| Cloud (Render) | Free | Easy | Excellent |

*Requires your Mac to stay on

---

## 🎯 Recommendation

**For testing/demos:** Use **ngrok** (easiest)
**For friends/family:** Use **Tailscale** (secure)
**For public access:** Use **cloud hosting** (Render/Railway)
**For learning:** Try **port forwarding + DDNS**

---

## 📝 Quick Commands Summary

```bash
# Start server
./start_server.sh

# Start ngrok (in another terminal)
ngrok http 8000

# Check if running
lsof -i :8000

# View logs
tail -f logs/stdout.log

# Stop server
# Ctrl+C or kill the process
```

