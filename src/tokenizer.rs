//! # Tokenizer Module
//!
//! This module converts text into numbers that a neural network can process.
//!
//! ## Why do we need a tokenizer?
//! Neural networks only understand numbers, not words. The tokenizer:
//! 1. Builds a vocabulary (a list of all known words)
//! 2. Assigns each word a unique ID (number)
//! 3. Converts sentences into sequences of these IDs
//!
//! ## Example:
//! ```text
//! Vocabulary: {"hello": 0, "world": 1, "how": 2, "are": 3, "you": 4}
//! "hello world" → [0, 1]
//! "how are you" → [2, 3, 4]
//! ```

use std::collections::HashMap;

/// A simple word-level tokenizer.
///
/// This tokenizer splits text on whitespace and converts each word to a unique ID.
/// Unknown words (not in vocabulary) get a special <UNK> token ID.
pub struct Tokenizer {
    /// Maps each word to its unique ID
    /// Example: {"win": 0, "money": 1, "now": 2, ...}
    word_to_id: HashMap<String, usize>,

    /// The ID assigned to unknown words (words not seen during training)
    /// This prevents crashes when we encounter new words during inference
    unk_id: usize,
}

impl Tokenizer {
    /// Creates a new tokenizer by building a vocabulary from the given texts.
    ///
    /// # How it works:
    /// 1. We iterate through all training texts
    /// 2. Split each text into words (by whitespace)
    /// 3. Add each unique word to our vocabulary with a unique ID
    /// 4. Reserve a special ID for unknown words (<UNK>)
    ///
    /// # Arguments
    /// * `texts` - A slice of training text strings
    ///
    /// # Returns
    /// A Tokenizer ready to convert text to token IDs
    pub fn new(texts: &[&str]) -> Self {
        // HashMap to store our vocabulary
        // Key: word (String), Value: unique ID (usize)
        let mut word_to_id: HashMap<String, usize> = HashMap::new();

        // Current ID to assign (starts at 0, increments for each new word)
        let mut current_id: usize = 0;

        // Build vocabulary from all training texts
        for text in texts {
            // Split text into words by whitespace and iterate
            // Example: "Win money now" → ["Win", "money", "now"]
            for word in text.split_whitespace() {
                // Convert to lowercase for consistency
                // "Win" and "win" should be the same word
                let word_lower = word.to_lowercase();

                // Only add if we haven't seen this word before
                // entry() gives us a way to insert only if key doesn't exist
                if !word_to_id.contains_key(&word_lower) {
                    word_to_id.insert(word_lower, current_id);
                    current_id += 1;
                }
            }
        }

        // Reserve the next ID for unknown words (<UNK>)
        // This is used when we encounter words not in our vocabulary
        let unk_id = current_id;

        Tokenizer { word_to_id, unk_id }
    }

    /// Converts a text string into a vector of token IDs.
    ///
    /// # How it works:
    /// 1. Split the text into words
    /// 2. Look up each word in our vocabulary
    /// 3. If found, use its ID; if not, use the <UNK> ID
    ///
    /// # Arguments
    /// * `text` - The text to tokenize
    ///
    /// # Returns
    /// A vector of token IDs representing the text
    ///
    /// # Example
    /// ```text
    /// tokenizer.encode("Win money now") → [0, 1, 2]
    /// tokenizer.encode("Unknown word here") → [UNK_ID, UNK_ID, UNK_ID]
    /// ```
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| {
                // Look up the word (lowercased) in our vocabulary
                // If not found, use the unknown token ID
                let word_lower = word.to_lowercase();
                *self.word_to_id.get(&word_lower).unwrap_or(&self.unk_id)
            })
            .collect()
    }

    /// Returns the size of the vocabulary (including the <UNK> token).
    ///
    /// This is needed to create the embedding layer, which needs to know
    /// how many unique tokens exist.
    ///
    /// # Returns
    /// The total vocabulary size = number of unique words + 1 (for <UNK>)
    pub fn vocab_size(&self) -> usize {
        // +1 because we have an additional <UNK> token
        self.word_to_id.len() + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_basic() {
        let texts = vec!["hello world", "hello rust"];
        let tokenizer = Tokenizer::new(&texts);

        // Should have 3 unique words + 1 UNK = vocab size 4
        assert_eq!(tokenizer.vocab_size(), 4);

        // Known words should get consistent IDs
        let encoded = tokenizer.encode("hello world");
        assert_eq!(encoded.len(), 2);
    }

    #[test]
    fn test_unknown_words() {
        let texts = vec!["hello"];
        let tokenizer = Tokenizer::new(&texts);

        // Unknown word should get the UNK ID
        let encoded = tokenizer.encode("unknown");
        assert_eq!(encoded[0], tokenizer.unk_id);
    }
}
