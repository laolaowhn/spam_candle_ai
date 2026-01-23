//! # Dataset Module
//!
//! This module contains our training data - examples of spam and ham messages.
//!
//! ## What is a Dataset?
//! A dataset is a collection of examples that we use to train our neural network.
//! Each example has:
//! - **Input**: The text message
//! - **Label**: What category it belongs to (0 = Ham, 1 = Spam)
//!
//! ## Why do we need labels?
//! Labels tell the neural network the "correct answer" during training.
//! The network learns by comparing its predictions to these labels and
//! adjusting its weights to get closer to the correct answers.
//!
//! ## Our Labels:
//! - **0 = Ham** (legitimate message, not spam)
//! - **1 = Spam** (unwanted promotional/scam message)

/// Represents a single training example.
///
/// Each example contains a text message and its corresponding label.
pub struct Example {
    /// The raw text of the message
    /// Example: "Win money now"
    pub text: &'static str,

    /// The label for this text
    /// - 0 = Ham (legitimate)
    /// - 1 = Spam (unwanted)
    pub label: u32,
}

/// Returns the hardcoded training dataset.
///
/// # Why more data?
/// The more diverse examples we have, the better the model generalizes.
/// With only a few examples, the model might memorize patterns instead
/// of learning general rules.
///
/// # Dataset Composition:
/// - Spam examples (label = 1): promotional, scam, urgency, "free", "win", "click"
/// - Ham examples (label = 0): greetings, work, daily conversation, "hello", "hi"
///
/// # Returns
/// A vector of Example structs containing our training data
pub fn get_dataset() -> Vec<Example> {
    vec![
        // ============================================
        // SPAM EXAMPLES (label = 1)
        // ============================================
        // Patterns: free offers, urgency, money, prizes, click bait
        Example {
            text: "Win money now",
            label: 1,
        },
        Example {
            text: "Free iPhone click here",
            label: 1,
        },
        Example {
            text: "Limited offer claim now",
            label: 1,
        },
        Example {
            text: "Congratulations you won a prize",
            label: 1,
        },
        Example {
            text: "Click here for free gift",
            label: 1,
        },
        Example {
            text: "Urgent action required immediately",
            label: 1,
        },
        Example {
            text: "You have been selected winner",
            label: 1,
        },
        Example {
            text: "Get rich quick scheme",
            label: 1,
        },
        Example {
            text: "Free money no strings attached",
            label: 1,
        },
        Example {
            text: "Act now limited time offer",
            label: 1,
        },
        Example {
            text: "Claim your free prize today",
            label: 1,
        },
        Example {
            text: "Win big cash rewards now",
            label: 1,
        },
        Example {
            text: "Click to claim free bonus",
            label: 1,
        },
        Example {
            text: "Exclusive offer just for you",
            label: 1,
        },
        Example {
            text: "You are our lucky winner",
            label: 1,
        },
        // ============================================
        // HAM EXAMPLES (label = 0)
        // ============================================
        // Normal greetings and conversation
        Example {
            text: "Hello how are you",
            label: 0,
        },
        Example {
            text: "Hello friend",
            label: 0,
        },
        Example {
            text: "Hello there",
            label: 0,
        },
        Example {
            text: "Hi how are you doing",
            label: 0,
        },
        Example {
            text: "Hi friend how is it going",
            label: 0,
        },
        Example {
            text: "Hey what is up",
            label: 0,
        },
        Example {
            text: "Good morning",
            label: 0,
        },
        Example {
            text: "Good morning friend",
            label: 0,
        },
        Example {
            text: "Good afternoon how are you",
            label: 0,
        },
        Example {
            text: "Good evening",
            label: 0,
        },
        // Work and daily life
        Example {
            text: "Let's meet tomorrow",
            label: 0,
        },
        Example {
            text: "Please review the document",
            label: 0,
        },
        Example {
            text: "See you at the meeting",
            label: 0,
        },
        Example {
            text: "Thanks for your help",
            label: 0,
        },
        Example {
            text: "Can we talk later today",
            label: 0,
        },
        Example {
            text: "The project looks great",
            label: 0,
        },
        Example {
            text: "Dinner at seven sounds good",
            label: 0,
        },
        Example {
            text: "Happy birthday to you",
            label: 0,
        },
        Example {
            text: "What time is the event",
            label: 0,
        },
        Example {
            text: "I will call you back",
            label: 0,
        },
        Example {
            text: "Nice to meet you",
            label: 0,
        },
        Example {
            text: "Have a great day",
            label: 0,
        },
        Example {
            text: "See you later",
            label: 0,
        },
        Example {
            text: "Take care friend",
            label: 0,
        },
        Example {
            text: "How is your day going",
            label: 0,
        },
    ]
}

/// Returns only the text strings from the dataset.
///
/// This is useful for building the tokenizer vocabulary,
/// which only needs the text, not the labels.
///
/// # Returns
/// A vector of text strings from all examples
pub fn get_texts() -> Vec<&'static str> {
    get_dataset().iter().map(|e| e.text).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_size() {
        let dataset = get_dataset();
        assert!(dataset.len() >= 25);
    }

    #[test]
    fn test_labels_valid() {
        let dataset = get_dataset();
        for example in dataset {
            assert!(example.label == 0 || example.label == 1);
        }
    }

    #[test]
    fn test_has_both_classes() {
        let dataset = get_dataset();
        let spam_count = dataset.iter().filter(|e| e.label == 1).count();
        let ham_count = dataset.iter().filter(|e| e.label == 0).count();
        assert!(spam_count >= 10);
        assert!(ham_count >= 10);
    }
}
