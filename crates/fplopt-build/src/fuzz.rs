//! Name matching for the Understat and ESPN maps: rapidfuzz's `fuzz.token_set_ratio`
//! (rapidfuzz 3, no processor) and `fplopt.build.understat.normalise_name`, ported exactly
//! (golden test over every name pair the build scores on real data:
//! `tests/fixtures/rust_golden/token_set_ratio.csv`).

use std::collections::BTreeSet;

use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::canonical_combining_class;

/// Indel distance (insertions + deletions) = len1 + len2 - 2 * LCS, over chars.
fn indel_distance(a: &[char], b: &[char]) -> usize {
    if a.is_empty() || b.is_empty() {
        return a.len() + b.len();
    }
    let mut previous = vec![0usize; b.len() + 1];
    let mut current = vec![0usize; b.len() + 1];
    for &ca in a {
        for (j, &cb) in b.iter().enumerate() {
            current[j + 1] =
                if ca == cb { previous[j] + 1 } else { previous[j + 1].max(current[j]) };
        }
        std::mem::swap(&mut previous, &mut current);
    }
    a.len() + b.len() - 2 * previous[b.len()]
}

fn norm_distance(dist: usize, lensum: usize) -> f64 {
    if lensum == 0 { 100.0 } else { 100.0 - 100.0 * dist as f64 / lensum as f64 }
}

/// `rapidfuzz.fuzz.token_set_ratio(s1, s2)`.
pub fn token_set_ratio(s1: &str, s2: &str) -> f64 {
    let tokens_a: BTreeSet<&str> = s1.split_whitespace().collect();
    let tokens_b: BTreeSet<&str> = s2.split_whitespace().collect();
    if tokens_a.is_empty() || tokens_b.is_empty() {
        return 0.0;
    }
    let intersect: Vec<&str> = tokens_a.intersection(&tokens_b).copied().collect();
    let diff_ab: Vec<&str> = tokens_a.difference(&tokens_b).copied().collect();
    let diff_ba: Vec<&str> = tokens_b.difference(&tokens_a).copied().collect();
    if !intersect.is_empty() && (diff_ab.is_empty() || diff_ba.is_empty()) {
        return 100.0;
    }
    let ab: Vec<char> = diff_ab.join(" ").chars().collect();
    let ba: Vec<char> = diff_ba.join(" ").chars().collect();
    let sect_len = intersect.join(" ").chars().count();
    let sep = usize::from(sect_len != 0);
    let sect_ab_len = sect_len + sep + ab.len();
    let sect_ba_len = sect_len + sep + ba.len();
    let result = norm_distance(indel_distance(&ab, &ba), sect_ab_len + sect_ba_len);
    if sect_len == 0 {
        return result;
    }
    let sect_ab_ratio = norm_distance(sep + ab.len(), sect_len + sect_ab_len);
    let sect_ba_ratio = norm_distance(sep + ba.len(), sect_len + sect_ba_len);
    result.max(sect_ab_ratio).max(sect_ba_ratio)
}

/// Python's `html.unescape` for the entities that occur in player names (numeric
/// references and the basic named ones).
pub fn html_unescape(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('&') {
        out.push_str(&rest[..start]);
        rest = &rest[start..];
        let end = rest.find(';').filter(|&e| e <= 32);
        let decoded = end.and_then(|e| {
            let body = &rest[1..e];
            let c = if let Some(hex) = body.strip_prefix("#x").or_else(|| body.strip_prefix("#X")) {
                u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
            } else if let Some(dec) = body.strip_prefix('#') {
                dec.parse::<u32>().ok().and_then(char::from_u32)
            } else {
                match body {
                    "amp" => Some('&'),
                    "lt" => Some('<'),
                    "gt" => Some('>'),
                    "quot" => Some('"'),
                    "apos" => Some('\''),
                    "nbsp" => Some('\u{a0}'),
                    _ => None,
                }
            };
            c.map(|c| (c, e))
        });
        match decoded {
            Some((c, e)) => {
                out.push(c);
                rest = &rest[e + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// HTML-unescaped, accents stripped/transliterated, lower case, letters/digits/spaces only.
pub fn normalise_name(name: &str) -> String {
    let lowered = html_unescape(name).to_lowercase();
    let mut text = String::with_capacity(lowered.len());
    for c in lowered.nfkd() {
        if canonical_combining_class(c) != 0 {
            continue;
        }
        match c {
            'ø' => text.push('o'),
            'æ' => text.push_str("ae"),
            'œ' => text.push_str("oe"),
            'ß' => text.push_str("ss"),
            'ł' => text.push('l'),
            'đ' | 'ð' => text.push('d'),
            'þ' => text.push_str("th"),
            'ı' => text.push('i'),
            c => text.push(c),
        }
    }
    let cleaned: String = text
        .chars()
        .map(|c| if c.is_ascii_lowercase() || c.is_ascii_digit() { c } else { ' ' })
        .collect();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rapidfuzz_doc_examples() {
        assert_eq!(token_set_ratio("fuzzy was a bear", "fuzzy fuzzy was a bear"), 100.0);
        assert_eq!(token_set_ratio("fuzzy was a bear but not a dog", "fuzzy was a bear"), 100.0);
        let score =
            token_set_ratio("fuzzy was a bear but not a dog", "fuzzy was a bear but not a cat");
        assert!((score - 92.3076923076923).abs() < 1e-9, "{score}");
        assert_eq!(token_set_ratio("", "x"), 0.0);
    }

    /// Python's results on every name the maps normalise and the pairs they score on real
    /// data (`dev/name_golden.py`): normalisation identical, scores within 1e-9.
    #[test]
    fn golden_names_and_pairs() {
        use std::io::Read;
        // FPLOPT_NAMES_GOLDEN: check another file (e.g. the untrimmed set, 1.8M pairs).
        let path = std::env::var_os("FPLOPT_NAMES_GOLDEN")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/fixtures/rust_golden/names.json.gz")
            });
        let mut text = String::new();
        flate2::read::GzDecoder::new(std::fs::File::open(path).unwrap())
            .read_to_string(&mut text)
            .unwrap();
        let golden: serde_json::Value = serde_json::from_str(&text).unwrap();
        for (name, expected) in golden["names"].as_object().unwrap() {
            assert_eq!(normalise_name(name), expected.as_str().unwrap(), "{name:?}");
        }
        let pairs = golden["pairs"].as_array().unwrap();
        assert!(pairs.len() > 40_000);
        for pair in pairs {
            let (a, b) = (pair[0].as_str().unwrap(), pair[1].as_str().unwrap());
            let expected = pair[2].as_f64().unwrap();
            let got = token_set_ratio(a, b);
            assert!((got - expected).abs() <= 1e-9, "{a:?} {b:?}: {got} vs {expected}");
        }
    }

    #[test]
    fn names_normalise() {
        assert_eq!(normalise_name("Martin Ødegaard"), "martin odegaard");
        assert_eq!(normalise_name("N&#039;Golo Kanté"), "n golo kante");
        assert_eq!(normalise_name("  Łukasz  Fabiański "), "lukasz fabianski");
    }
}
