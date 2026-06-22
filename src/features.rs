//! Feature extraction for Persian NER.
//!
//! Extends the POS feature set with character n-grams (4/5-gram) and
//! orthographic flags relevant to Persian named entities.

fn all_digits(word: &str) -> bool {
    !word.is_empty() && word.chars().all(|c| c.is_ascii_digit() || matches!(c, '۰'..='۹'))
}

fn is_punc(word: &str) -> bool {
    const PUNC: &[&str] = &[".", ",", "!", "?", "؟", "،", ":", ";", "(", ")", "[", "]", "«", "»", "-"];
    PUNC.contains(&word)
}

/// Extracts NER features for `sentence[index]`.
pub fn ner_features(sentence: &[&str], index: usize) -> Vec<String> {
    let word = sentence[index];
    let n = sentence.len();
    let prev  = if index > 0 { sentence[index - 1] } else { "" };
    let prev2 = if index > 1 { sentence[index - 2] } else { "" };
    let next  = if index + 1 < n { sentence[index + 1] } else { "" };
    let next2 = if index + 2 < n { sentence[index + 2] } else { "" };

    let chars: Vec<char> = word.chars().collect();
    let len = chars.len();

    let prefix = |k: usize| -> String { chars.iter().take(k).collect() };
    let suffix = |k: usize| -> String {
        if len >= k {
            chars[len - k..].iter().collect()
        } else {
            chars.iter().collect()
        }
    };

    let mut f = vec![
        // Word identity
        format!("word={}", word),
        // Prefixes up to 4 chars
        format!("pfx1={}", prefix(1)),
        format!("pfx2={}", prefix(2)),
        format!("pfx3={}", prefix(3)),
        format!("pfx4={}", prefix(4)),
        // Suffixes up to 4 chars
        format!("sfx1={}", suffix(1)),
        format!("sfx2={}", suffix(2)),
        format!("sfx3={}", suffix(3)),
        format!("sfx4={}", suffix(4)),
        // Context
        format!("prev={}", prev),
        format!("prev2={}", prev2),
        format!("next={}", next),
        format!("next2={}", next2),
        // Bigram context
        format!("prev+word={}+{}", prev, word),
        format!("word+next={}+{}", word, next),
        // Orthographic
        format!("is_digit={}", all_digits(word)),
        format!("is_punc={}", is_punc(word)),
        format!("word_len={}", len.min(10)),
    ];

    if index == 0 { f.push("BOS".to_string()); }
    if index == n - 1 { f.push("EOS".to_string()); }

    f
}

/// Converts a full sentence into per-token NER feature lists.
pub fn sentence_ner_features(sentence: &[&str]) -> Vec<Vec<String>> {
    (0..sentence.len()).map(|i| ner_features(sentence, i)).collect()
}
