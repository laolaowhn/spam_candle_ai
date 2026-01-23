# 📧 Spam vs Ham Text Classifier

A beginner-friendly machine learning project in **pure Rust** using the [Candle](https://github.com/huggingface/candle) ML framework.

> No Python. No external ML libraries. Just Rust. 🦀

## 🎯 What It Does

Classifies text messages as **Spam** or **Ham** (legitimate) using a neural network.

```
🚫 "Win money now"        → SPAM (100%)
✅ "Hello friend"         → HAM  (100%)
🚫 "Free iPhone click"    → SPAM (100%)
✅ "See you tomorrow"     → HAM  (100%)
```

## 🧠 Neural Network Architecture

```
Input Text → Tokenizer → Embedding → Mean Pooling → Linear → Softmax → Prediction
                            ↓            ↓            ↓         ↓
                        [N, 16]        [16]         [2]       [2]
```

| Layer | Purpose |
|-------|---------|
| **Tokenizer** | Converts words to numbers |
| **Embedding** | Maps word IDs to dense vectors |
| **Mean Pooling** | Averages vectors to fixed size |
| **Linear** | Classification layer |
| **Softmax** | Converts to probabilities |

## 📁 Project Structure

```
spam_candle_ai/
├── Cargo.toml              # Dependencies
├── src/
│   ├── main.rs             # Training loop + inference
│   ├── model.rs            # Neural network definition
│   ├── tokenizer.rs        # Word tokenizer
│   └── dataset.rs          # Training data (40 examples)
└── spam_classifier.safetensors  # Saved model (after training)
```

## 🚀 Quick Start

```bash
# Clone and run
cd spam_candle_ai
cargo run
```

## 📊 Training Output

```
Training for 300 epochs...
----------------------------------------------
Epoch   1 | Average Loss: 0.5388
Epoch 100 | Average Loss: 0.0022
Epoch 200 | Average Loss: 0.0009
Epoch 300 | Average Loss: 0.0005
----------------------------------------------
✓ Training complete!
✓ Model saved to 'spam_classifier.safetensors'
```

## 🎮 Interactive Mode

After training, type your own messages to classify:

```
Enter text: hello friend
✅ "hello friend"         → HAM  (confidence: 100.0%)

Enter text: click here free money
🚫 "click here free money" → SPAM (confidence: 99.9%)

Enter text: quit
Goodbye! 👋
```

## 📦 Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `candle-core` | 0.9 | Tensor operations |
| `candle-nn` | 0.9 | Neural network layers |
| `anyhow` | 1.0 | Error handling |

## 🎓 Learning Resources

This project is designed for Rust developers learning ML. Key concepts:

1. **Tokenization** - Converting text to numbers
2. **Embeddings** - Learning word representations
3. **Forward Pass** - Running data through the model
4. **Loss Function** - Measuring prediction error
5. **Backpropagation** - Computing gradients
6. **SGD Optimizer** - Updating weights

## 📝 License

MIT

---

Made with ❤️ in Rust 🦀
