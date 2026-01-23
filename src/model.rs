//! # Model Module
//!
//! This module defines our neural network for spam classification.
//!
//! ## Neural Network Architecture
//! ```text
//! Input Token IDs → Embedding → Mean Pooling → Linear → Output
//!      [N]           [N, D]        [D]          [2]
//! ```
//!
//! Where:
//! - N = Number of tokens in the input text
//! - D = Embedding dimension (size of each word's vector representation)
//! - 2 = Number of output classes (Ham, Spam)
//!
//! ## What does each layer do?
//!
//! ### 1. Embedding Layer
//! Converts each token ID into a dense vector of numbers.
//! Think of it as looking up each word in a "meaning dictionary"
//! where each word has D numbers describing its meaning.
//!
//! ### 2. Mean Pooling
//! Averages all word vectors into a single vector.
//! This converts variable-length input (different sentence lengths)
//! into a fixed-size representation.
//!
//! ### 3. Linear Layer
//! The classification layer that takes the pooled representation
//! and outputs a score for each class (Ham or Spam).

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder, VarMap};

/// The spam classifier neural network.
///
/// This struct holds all the learnable parameters of our model:
/// - An embedding table (vocab_size × embedding_dim matrix)
/// - A linear layer (embedding_dim × 2 matrix + bias)
pub struct SpamClassifier {
    /// Embedding layer: converts token IDs to vectors
    /// Shape: [vocab_size, embedding_dim]
    /// Each row is a learnable vector for one word
    embedding: Embedding,

    /// Linear classification layer
    /// Weights shape: [embedding_dim, 2]
    /// Bias shape: [2]
    linear: Linear,
}

impl SpamClassifier {
    /// Creates a new SpamClassifier with random weights.
    ///
    /// # Arguments
    /// * `vocab_size` - Number of unique tokens in our vocabulary
    /// * `embedding_dim` - Size of each word's vector representation
    /// * `var_map` - Variable map to store learnable parameters
    /// * `device` - Device to run on (CPU in our case)
    ///
    /// # How it works
    /// 1. Creates an embedding table of size [vocab_size, embedding_dim]
    /// 2. Creates a linear layer mapping [embedding_dim] → [2]
    /// 3. All weights are randomly initialized
    ///
    /// # Returns
    /// A new SpamClassifier ready for training
    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        var_map: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        // VarBuilder helps us create and track learnable parameters
        // All variables created through it will be stored in var_map
        let vb = VarBuilder::from_varmap(var_map, DType::F32, device);

        // Create the embedding layer
        // - vocab_size: how many words we have
        // - embedding_dim: how many numbers describe each word
        // - "embed": name for this layer's variables (for debugging)
        let embedding = embedding(vocab_size, embedding_dim, vb.pp("embed"))?;

        // Create the linear layer
        // - embedding_dim: input size (from mean pooling)
        // - 2: output size (two classes: Ham and Spam)
        // - "classifier": name for this layer's variables
        let linear = linear(embedding_dim, 2, vb.pp("classifier"))?;

        Ok(Self { embedding, linear })
    }

    /// Forward pass: converts token IDs to class probabilities.
    ///
    /// # The Forward Pass Step by Step:
    ///
    /// ## Step 1: Embedding Lookup
    /// ```text
    /// Input: [token_id_1, token_id_2, ..., token_id_N]
    /// Output: [[vec_1], [vec_2], ..., [vec_N]]
    /// Shape: [N] → [N, embedding_dim]
    /// ```
    /// Each token ID is replaced with its embedding vector.
    ///
    /// ## Step 2: Mean Pooling
    /// ```text
    /// Input: [[vec_1], [vec_2], ..., [vec_N]]
    /// Output: [mean_vec]
    /// Shape: [N, embedding_dim] → [embedding_dim]
    /// ```
    /// Average all vectors to get one fixed-size representation.
    ///
    /// ## Step 3: Linear Classification
    /// ```text
    /// Input: [mean_vec]
    /// Output: [ham_score, spam_score]
    /// Shape: [embedding_dim] → [2]
    /// ```
    /// The linear layer produces a score for each class.
    ///
    /// # Arguments
    /// * `token_ids` - Tensor of token IDs, shape [N]
    ///
    /// # Returns
    /// Raw logits (scores) for each class, shape [2]
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        // Step 1: Embedding lookup
        // Input shape: [N] where N = number of tokens
        // Output shape: [N, embedding_dim]
        let embeddings = self.embedding.forward(token_ids)?;

        // Step 2: Mean pooling - average across the sequence dimension
        // We average over dimension 0 (the N tokens)
        // Input shape: [N, embedding_dim]
        // Output shape: [embedding_dim] (after squeeze)
        //
        // Why mean pooling?
        // - Different sentences have different lengths
        // - We need a fixed-size vector for the classifier
        // - Averaging captures the "overall meaning" of all words
        let pooled = embeddings.mean(0)?;

        // Step 3: Linear layer - project to class scores
        // Input shape: [embedding_dim]
        // Output shape: [2] (one score per class)
        //
        // We need to add a batch dimension for the linear layer
        // [embedding_dim] → [1, embedding_dim] → linear → [1, 2] → [2]
        let pooled_batched = pooled.unsqueeze(0)?;
        let logits = self.linear.forward(&pooled_batched)?;
        let logits = logits.squeeze(0)?;

        Ok(logits)
    }
}
