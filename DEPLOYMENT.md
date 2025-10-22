# Deployment Guide for Phonetic Analysis Studio

## 🚀 Quick Deploy to Render (Recommended - FREE)

### Prerequisites:
1. GitHub account
2. Google Gemini API key

### Steps:

1. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
gh repo create phonetic-analysis --public --source=. --remote=origin
git push -u origin main
```

2. **Deploy to Render:**
   - Go to https://render.com and sign up
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `phonetic-analysis`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Click "Advanced" and add environment variable:
     - **Key**: `GEMINI_API_KEY`
     - **Value**: `your-gemini-api-key`
   - Click "Create Web Service"

3. **Wait for deployment** (5-10 minutes)

4. **Access your app** at: `https://your-app-name.onrender.com`

---

## 🚂 Alternative: Deploy to Railway

1. **Push to GitHub** (same as above)

2. **Deploy to Railway:**
   - Go to https://railway.app
   - Click "Start a New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Add environment variable:
     - `GEMINI_API_KEY` = your API key
   - Railway will auto-detect and deploy

3. **Access your app** at the Railway-provided URL

---

## ⚙️ Environment Variables

Make sure to set these in your hosting platform:

- `GEMINI_API_KEY`: Your Google Gemini API key
- `PORT`: (Usually auto-set by the platform)

---

## 📦 Files for Deployment

The following files have been created for deployment:

- `Procfile`: Tells hosting platforms how to run your app
- `runtime.txt`: Specifies Python version
- `railway.json`: Configuration for Railway
- `requirements.txt`: Python dependencies

---

## 🔧 Troubleshooting

### Issue: App crashes on startup
**Solution**: Check logs for missing dependencies, add them to `requirements.txt`

### Issue: Audio not working
**Solution**: Ensure your deployment platform supports WebRTC and audio processing

### Issue: High memory usage
**Solution**: 
- Reduce model size in production
- Use CPU instead of GPU for inference
- Add memory limits in deployment config

### Issue: Slow first request
**Solution**: Most free hosting platforms have cold starts. Consider:
- Using a paid tier for always-on instances
- Implementing lazy loading for heavy models
- Adding a health check endpoint

---

## 💡 Production Tips

1. **Security**:
   - Never commit API keys to Git
   - Use environment variables for secrets
   - Add rate limiting for production

2. **Performance**:
   - Consider using a CDN for static files
   - Implement caching for common requests
   - Monitor memory usage

3. **Monitoring**:
   - Set up error logging
   - Monitor API usage and costs
   - Track user sessions

---

## 🌐 Custom Domain (Optional)

Most platforms allow custom domains:

**Render:**
- Go to Settings → Custom Domain
- Add your domain
- Update DNS records as instructed

**Railway:**
- Click on your service → Settings → Domains
- Add custom domain
- Update DNS with provided CNAME

---

## 📊 Cost Estimates

- **Render Free Tier**: 
  - Free with limitations (sleeps after inactivity)
  - Good for testing and demos

- **Railway**: 
  - $5/month for basic usage
  - Pay-as-you-go for resources

- **Heroku**: 
  - Starting at $7/month
  - No free tier

- **DigitalOcean**: 
  - Starting at $5/month
  - More predictable pricing

---

## 🎯 Next Steps After Deployment

1. Test all features on the live site
2. Monitor logs for errors
3. Set up analytics (optional)
4. Share your app URL!

---

## 📝 Quick Commands Reference

```bash
# Initialize Git
git init
git add .
git commit -m "Initial commit"

# Create GitHub repo (using gh CLI)
gh repo create phonetic-analysis --public --source=. --remote=origin
git push -u origin main

# Or manually:
# 1. Create repo on github.com
# 2. git remote add origin <your-repo-url>
# 3. git push -u origin main
```

---

## 🆘 Support

If you encounter issues:
1. Check the platform's documentation
2. Review deployment logs
3. Verify environment variables are set
4. Ensure all dependencies are in requirements.txt

