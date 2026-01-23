//! # Spam vs Ham Text Classifier
//!
//! A beginner-friendly machine learning project in pure Rust using Candle.
//!
//! ## What this project does:
//! 1. Takes text messages as input
//! 2. Classifies them as either Spam or Ham (legitimate)
//!
//! ## How machine learning works (simplified):
//!
//! ### Training Phase:
//! 1. Show the model many examples of spam and ham messages
//! 2. For each example, the model makes a prediction
//! 3. Compare prediction to the correct answer (the label)
//! 4. Calculate how wrong the prediction was (the "loss")
//! 5. Adjust the model's weights to reduce the loss
//! 6. Repeat many times (epochs) until the model learns
//!
//! ### Inference Phase:
//! 1. Give the trained model a new message
//! 2. The model outputs probabilities for each class
//! 3. Pick the class with highest probability

mod dataset;
mod model;
mod tokenizer;

use std::io::{self, Write};

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{optim::Optimizer, VarMap};

use dataset::{get_dataset, get_texts};
use model::SpamClassifier;
use tokenizer::Tokenizer;

/// Learning rate controls how big our weight updates are.
///
/// - Too high: Training is unstable, loss jumps around
/// - Too low: Training is very slow
/// - Just right: Loss decreases steadily
///
/// 0.1 is a reasonable starting point for small models.
const LEARNING_RATE: f64 = 0.1;

/// Embedding dimension - size of each word's vector representation.
///
/// - Larger = more expressive, but needs more data to train
/// - Smaller = simpler, works well with small datasets
///
/// 16 is sufficient for our tiny dataset.
const EMBEDDING_DIM: usize = 16;

/// Number of training epochs (passes through the dataset).
///
/// One epoch = seeing every training example once.
/// More epochs = more learning opportunities.
/// With 30 examples, 300 epochs gives good results.
const EPOCHS: usize = 300;

/// Path to save the trained model weights
const MODEL_PATH: &str = "spam_classifier.safetensors";

fn main() -> Result<()> {
    println!("==============================================");
    println!("   Spam vs Ham Classifier - Candle ML");
    println!("==============================================\n");

    // ========================================
    // How it works (summary)
    // ========================================
    println!("How it works:");
    println!("  1. Tokenizer converts text → numbers");
    println!("  2. Embedding layer converts numbers → vectors");
    println!("  3. Mean pooling averages vectors → single vector");
    println!("  4. Linear layer classifies → Ham/Spam scores");
    println!("  5. Softmax converts scores → probabilities\n");

    // ========================================
    // STEP 1: Setup
    // ========================================
    // We use CPU for this tutorial (no GPU required)
    let device = Device::Cpu;

    // ========================================
    // STEP 2: Build Tokenizer
    // ========================================
    // The tokenizer converts text to numbers that the neural network can process
    println!("Building vocabulary from training data...");

    // Get all text strings from our dataset
    let texts = get_texts();

    // Create a tokenizer that builds a vocabulary from these texts
    let tokenizer = Tokenizer::new(&texts);

    println!("✓ Vocabulary size: {} words\n", tokenizer.vocab_size());

    // ========================================
    // STEP 3: Prepare Training Data
    // ========================================
    // Convert all our text examples to tensors (multi-dimensional arrays)
    println!("Preparing training data...");

    let dataset = get_dataset();

    // We'll store tokenized inputs and labels
    let mut training_data: Vec<(Tensor, u32)> = Vec::new();

    for example in &dataset {
        // Convert text to token IDs
        // "Win money now" → [0, 1, 2]
        let token_ids = tokenizer.encode(example.text);

        // Convert usize to u32 (required by Candle)
        let token_ids_u32: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();

        // Convert to a Tensor (Candle's array type)
        // Shape: [N] where N = number of tokens
        let token_tensor = Tensor::new(token_ids_u32.as_slice(), &device)?;

        training_data.push((token_tensor, example.label));
    }

    println!("✓ Prepared {} training examples\n", training_data.len());

    // ========================================
    // STEP 4: Create the Model
    // ========================================
    // Initialize our neural network with random weights
    println!("Initializing model...");

    // VarMap tracks all learnable parameters
    // This is needed for computing gradients and updating weights
    let var_map = VarMap::new();

    // Create the classifier
    let model = SpamClassifier::new(
        tokenizer.vocab_size(), // Input vocabulary size
        EMBEDDING_DIM,          // Size of word embeddings
        &var_map,               // Where to store parameters
        &device,                // Which device (CPU)
    )?;

    println!(
        "✓ Model initialized with {} embedding dimensions\n",
        EMBEDDING_DIM
    );

    // ========================================
    // STEP 5: Create Optimizer
    // ========================================
    // SGD (Stochastic Gradient Descent) is a simple optimizer
    // It updates weights by moving in the opposite direction of gradients
    let mut optimizer = candle_nn::optim::SGD::new(var_map.all_vars(), LEARNING_RATE)?;

    // ========================================
    // STEP 6: Training Loop
    // ========================================
    println!("Training for {} epochs...", EPOCHS);
    println!("----------------------------------------------");

    for epoch in 1..=EPOCHS {
        // Track total loss for this epoch
        let mut total_loss = 0.0;

        for (token_tensor, label) in &training_data {
            // ----- Forward Pass -----
            // Run the input through the model to get predictions

            // Get raw scores (logits) for each class
            // Shape: [2] - one score for Ham, one for Spam
            let logits = model.forward(token_tensor)?;

            // Apply softmax to convert logits to probabilities
            // Softmax makes all values positive and sum to 1
            // Shape: [2] - probabilities for [Ham, Spam]
            let probs = candle_nn::ops::softmax(&logits, 0)?;

            // ----- Compute Loss -----
            // Cross-entropy loss measures how wrong our prediction is

            // Get the probability assigned to the correct class
            // If label=1 (Spam), we want probs[1] to be high
            let target_prob = probs.get(*label as usize)?;

            // Cross-entropy loss = -log(probability of correct class)
            // - If we predict 0.99 for correct class: loss = -log(0.99) ≈ 0.01 (good!)
            // - If we predict 0.01 for correct class: loss = -log(0.01) ≈ 4.6 (bad!)
            let loss = target_prob.log()?.neg()?;

            // Accumulate loss for reporting
            total_loss += loss.to_scalar::<f32>()?;

            // ----- Backward Pass + Weight Update -----
            // The optimizer computes gradients and updates weights in one step
            // backward_step does: loss.backward() + apply SGD update
            optimizer.backward_step(&loss)?;
        }

        // Print progress every 10 epochs
        if epoch % 10 == 0 || epoch == 1 {
            let avg_loss = total_loss / training_data.len() as f32;
            println!("Epoch {:3} | Average Loss: {:.4}", epoch, avg_loss);
        }
    }

    println!("----------------------------------------------");
    println!("✓ Training complete!\n");

    // ========================================
    // STEP 7: Save the Model
    // ========================================
    // Save the trained weights to a file so we can load them later
    // SafeTensors is a safe, fast format for storing model weights
    println!("Saving model to '{}'...", MODEL_PATH);
    var_map.save(MODEL_PATH)?;
    println!("✓ Model saved successfully!\n");

    // ========================================
    // STEP 8: Test Predictions
    // ========================================
    // Test the trained model on various inputs
    println!("==============================================");
    println!("   Testing the trained model");
    println!("==============================================\n");

    // Test examples - mix of training data and new unseen messages
    let test_texts = vec![
        // Examples from training data (should be confident)
        "Win money now",
        "Let's meet tomorrow",
        "Free iPhone click here",
        "How are you doing",
        // New unseen examples (testing generalization)
        "Free gift for you",
        "Dinner at seven",
        "Click here to win",
        "See you at the meeting",
    ];

    for text in test_texts {
        predict_text(&model, &tokenizer, &device, text)?;
    }

    // ========================================
    // STEP 9: Interactive Mode
    // ========================================
    // Let the user type their own messages to classify
    println!("\n==============================================");
    println!("   Interactive Mode");
    println!("==============================================");
    println!("Type a message and press Enter to classify it.");
    println!("Type 'quit' or 'exit' to stop.\n");

    loop {
        // Print prompt
        print!("Enter text: ");
        io::stdout().flush()?; // Make sure prompt is displayed

        // Read user input
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        // Trim whitespace
        let input = input.trim();

        // Check for exit commands
        if input.is_empty() {
            continue;
        }
        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            println!("\nGoodbye! 👋");
            break;
        }

        // Make prediction
        predict_text(&model, &tokenizer, &device, input)?;
    }

    Ok(())
}

/// Helper function to make a prediction on a single text
///
/// # Arguments
/// * `model` - The trained SpamClassifier
/// * `tokenizer` - The tokenizer to convert text to tokens
/// * `device` - The device (CPU) for tensor operations
/// * `text` - The text to classify
///
/// # Returns
/// Result indicating success or failure
fn predict_text(
    model: &SpamClassifier,
    tokenizer: &Tokenizer,
    device: &Device,
    text: &str,
) -> Result<()> {
    // Tokenize the input
    let token_ids = tokenizer.encode(text);
    let token_ids_u32: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();
    let token_tensor = Tensor::new(token_ids_u32.as_slice(), device)?;

    // Run inference (no gradient tracking needed)
    let logits = model.forward(&token_tensor)?;
    let probs = candle_nn::ops::softmax(&logits, 0)?;

    // Get probabilities for each class
    let ham_prob = probs.get(0)?.to_scalar::<f32>()?;
    let spam_prob = probs.get(1)?.to_scalar::<f32>()?;

    // Determine prediction
    let (prediction, confidence) = if spam_prob > ham_prob {
        ("SPAM", spam_prob)
    } else {
        ("HAM ", ham_prob)
    };

    // Display result with emoji for fun!
    let emoji = if prediction == "SPAM" { "🚫" } else { "✅" };
    println!(
        "{} {:30} → {} (confidence: {:.1}%)",
        emoji,
        format!("\"{}\"", text),
        prediction,
        confidence * 100.0
    );

    Ok(())
}
