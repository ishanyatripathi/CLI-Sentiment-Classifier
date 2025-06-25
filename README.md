
🧠 CLI Sentiment Classifier with LangGraph

This project is a simple and interactive **Command-Line Sentiment Classifier** that classifies user reviews as **positive** or **negative**, with a **fallback mechanism** for low-confidence predictions.

It was built as part of an internship task and combines:

- 🔍 DistilBERT for sentiment classification
- 🔁 LangGraph to build dynamic execution graphs
- 🤝 User feedback to correct low-confidence predictions
- 📝 Logging of all interactions to a file

---

📁 Folder Structure

task3/
├── nodes/
│   ├── inference_node.py
│   ├── check_confidence_node.py
│   └── fallback_node.py
├── sentiment-model/            ← trained DistilBERT model saved here
├── logs/
│   └── chatlog.txt             ← interaction logs
├── main.py                     ← main CLI pipeline using LangGraph
└── train_sentiment.py          ← fine-tunes and saves the model

---

🚀 How to Run

1. Clone the Repository

git clone https://github.com/<your-username>/chatbot.git
cd chatbot

2. Install Required Packages

Ensure Python 3.10+ is installed.

pip install transformers datasets torch langgraph

3. (Optional) Fine-Tune the Sentiment Model

python train_sentiment.py

This trains the model on a subset of the Yelp Polarity dataset and saves it to:

./sentiment-model/

4. Start the Chat Classifier

python main.py

Example:

🤖 Sentiment Classifier CLI (type /exit to quit)

📝 Enter your review: movie was worse
⚠️  Low confidence (64.8%). Asking for user confirmation.
🤔 Is the prediction "negative" correct? (y/n): y

✅ Final Label: negative (Confidence: 64.8%)

---

🧠 LangGraph Architecture

This classifier uses a stateful graph with four nodes:

| Node               | Purpose                                     |
|--------------------|---------------------------------------------|
| inference          | Predicts sentiment using DistilBERT         |
| check_confidence   | Checks if the prediction confidence ≥ 75%   |
| fallback           | If confidence is low, asks for user feedback|
| log                | Logs the result to logs/chatlog.txt         |

Flow Diagram:
[inference] → [check_confidence] → [fallback or log] → END

---

💡 Features

- ✅ Built using LangGraph's StateGraph abstraction
- ✅ Low-confidence fallback mechanism with user correction
- ✅ Interaction logs saved with timestamps
- ✅ Modular design with reusable nodes
- ✅ Easily extendable to web or GUI interfaces

---

🎥 Introduction video: https://drive.google.com/file/d/1qQ5mpU0d51-y7NUn3PFCMqwDQ_a-EqxT/view?usp=sharing

📄 License

This project is submitted as part of an internship assignment and is intended for educational demonstration purposes only.
