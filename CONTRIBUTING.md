# Contributing to SkyGuard

Thank you for your interest in contributing to SkyGuard! 

## 🔧 Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/skyguard.git
   cd skyguard
   ```

2. **Set up environment**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Add your API keys to .env file
   # Get Gemini API key from: https://makersuite.google.com/app/apikey
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app/main_app.py
   ```

## 📋 How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

## 🧪 Testing

Before submitting a PR, please run:
```bash
# Test the main application
streamlit run app/main_app.py
```

## 📝 Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and small

## 🐛 Reporting Issues

Please use the GitHub Issues tab to report bugs or request features.

## 📞 Contact

For questions, please open an issue or reach out to the maintainers.
