//! Forced splits (ADV-03) — a JSON-driven forced split structure applied at the
//! TOP of tree growth, before the gain argmax (`serial_tree_learner.cpp:620-734`
//! `ForceSplits`). When no forced-splits JSON is configured this module is never
//! consulted and the spine growth is byte-untouched (D-06).
//!
//! ## Security (T-07-11-01, Tampering/DoS)
//! `forced_splits_filename` is the only UNTRUSTED input crossing the crate
//! boundary in this plan. The parser:
//! - hand-rolls a SMALL recursive-descent JSON reader for exactly the
//!   `{feature, threshold, left, right}` schema (NO new external dependency — the
//!   Package Legitimacy Gate is therefore not triggered);
//! - validates the schema + bounds (feature index in range, threshold finite,
//!   nesting depth bounded) and returns a typed [`ForcedSplitError`] on malformed
//!   input — never a panic;
//! - bounds the nesting depth (`MAX_FORCED_DEPTH`) so a deeply-nested adversarial
//!   document cannot blow the stack or grow an unbounded tree.

use std::fmt;

/// Maximum forced-split nesting depth (DoS guard, T-07-11-01). A forced-split
/// tree deeper than this is rejected as malformed.
pub const MAX_FORCED_DEPTH: usize = 64;

/// One forced-split node: a feature/threshold plus optional left/right children
/// (`{ "feature": int, "threshold": double, "left": {...}, "right": {...} }`).
#[derive(Debug, Clone, PartialEq)]
pub struct ForcedSplitNode {
    /// REAL feature index to force a split on at this node.
    pub feature: i32,
    /// The real-value threshold (`BinThreshold` maps it to a bin at apply time).
    pub threshold: f64,
    /// The forced sub-tree for the LEFT child (rows `<= threshold`), if any.
    pub left: Option<Box<ForcedSplitNode>>,
    /// The forced sub-tree for the RIGHT child (rows `> threshold`), if any.
    pub right: Option<Box<ForcedSplitNode>>,
}

/// A typed error for a malformed forced-splits document (never a panic).
#[derive(Debug, Clone, PartialEq)]
pub enum ForcedSplitError {
    /// The JSON could not be tokenized/parsed (syntax error at byte offset).
    Syntax(String),
    /// A required key (`feature` / `threshold`) was missing on a forced node.
    MissingKey(&'static str),
    /// A value had the wrong type (e.g. `feature` was not an integer).
    WrongType(&'static str),
    /// The feature index was out of range `[0, num_features)`.
    FeatureOutOfRange { feature: i32, num_features: i32 },
    /// The threshold was non-finite (NaN/Inf).
    NonFiniteThreshold,
    /// The forced-split nesting exceeded [`MAX_FORCED_DEPTH`].
    TooDeep,
}

impl fmt::Display for ForcedSplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ForcedSplitError::Syntax(s) => write!(f, "forced_splits JSON syntax error: {s}"),
            ForcedSplitError::MissingKey(k) => write!(f, "forced_splits node missing key `{k}`"),
            ForcedSplitError::WrongType(k) => write!(f, "forced_splits key `{k}` has wrong type"),
            ForcedSplitError::FeatureOutOfRange { feature, num_features } => write!(
                f,
                "forced_splits feature {feature} out of range [0, {num_features})"
            ),
            ForcedSplitError::NonFiniteThreshold => {
                write!(f, "forced_splits threshold must be finite")
            }
            ForcedSplitError::TooDeep => {
                write!(f, "forced_splits nesting exceeds MAX_FORCED_DEPTH ({MAX_FORCED_DEPTH})")
            }
        }
    }
}

impl std::error::Error for ForcedSplitError {}

/// Parse + validate a forced-splits JSON document into a [`ForcedSplitNode`] tree,
/// checking every feature index against `num_features` and bounding the nesting
/// depth. Returns `Ok(None)` for an empty/whitespace document (no forced split).
///
/// # Errors
/// Returns [`ForcedSplitError`] on any syntax / schema / bounds violation.
pub fn parse_forced_splits(
    src: &str,
    num_features: i32,
) -> Result<Option<ForcedSplitNode>, ForcedSplitError> {
    let trimmed = src.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let mut p = JsonParser::new(trimmed);
    let value = p.parse_value(0)?;
    p.skip_ws();
    if !p.at_end() {
        return Err(ForcedSplitError::Syntax("trailing characters".to_string()));
    }
    let node = value_to_node(&value, num_features, 0)?;
    Ok(Some(node))
}

/// A minimal JSON value (object/number/string/null) — the only shapes the
/// forced-splits schema uses. Booleans are tolerated as scalar values but not
/// expected in the schema (their payload is intentionally unread — the schema
/// only reads `feature`/`threshold` numbers + nested objects).
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum JsonValue {
    Object(Vec<(String, JsonValue)>),
    Number(f64),
    String(String),
    Bool(bool),
    Null,
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            bytes: src.as_bytes(),
            pos: 0,
        }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue, ForcedSplitError> {
        if depth > MAX_FORCED_DEPTH {
            return Err(ForcedSplitError::TooDeep);
        }
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(depth),
            Some(b'"') => Ok(JsonValue::String(self.parse_string()?)),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'n') => self.parse_null(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(ForcedSplitError::Syntax(format!(
                "unexpected character `{}`",
                c as char
            ))),
            None => Err(ForcedSplitError::Syntax("unexpected end of input".to_string())),
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue, ForcedSplitError> {
        self.pos += 1; // consume '{'
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(items));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(ForcedSplitError::Syntax("expected object key string".to_string()));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(ForcedSplitError::Syntax("expected `:` after key".to_string()));
            }
            self.pos += 1;
            let val = self.parse_value(depth + 1)?;
            items.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(ForcedSplitError::Syntax("expected `,` or `}`".to_string())),
            }
        }
        Ok(JsonValue::Object(items))
    }

    fn parse_string(&mut self) -> Result<String, ForcedSplitError> {
        self.pos += 1; // consume opening quote
        let mut s = String::new();
        loop {
            match self.peek() {
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(s);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    match self.peek() {
                        Some(b'"') => s.push('"'),
                        Some(b'\\') => s.push('\\'),
                        Some(b'/') => s.push('/'),
                        Some(b'n') => s.push('\n'),
                        Some(b't') => s.push('\t'),
                        Some(b'r') => s.push('\r'),
                        _ => return Err(ForcedSplitError::Syntax("bad escape".to_string())),
                    }
                    self.pos += 1;
                }
                Some(c) => {
                    s.push(c as char);
                    self.pos += 1;
                }
                None => return Err(ForcedSplitError::Syntax("unterminated string".to_string())),
            }
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue, ForcedSplitError> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == b'-' || c == b'+' || c == b'.' || c == b'e' || c == b'E' || c.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        let slice = &self.bytes[start..self.pos];
        let s = std::str::from_utf8(slice)
            .map_err(|_| ForcedSplitError::Syntax("invalid number bytes".to_string()))?;
        s.parse::<f64>()
            .map(JsonValue::Number)
            .map_err(|_| ForcedSplitError::Syntax(format!("invalid number `{s}`")))
    }

    fn parse_bool(&mut self) -> Result<JsonValue, ForcedSplitError> {
        if self.bytes[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(JsonValue::Bool(true))
        } else if self.bytes[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(JsonValue::Bool(false))
        } else {
            Err(ForcedSplitError::Syntax("invalid literal".to_string()))
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, ForcedSplitError> {
        if self.bytes[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(JsonValue::Null)
        } else {
            Err(ForcedSplitError::Syntax("invalid literal".to_string()))
        }
    }
}

fn object_get<'a>(items: &'a [(String, JsonValue)], key: &str) -> Option<&'a JsonValue> {
    items.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn value_to_node(
    value: &JsonValue,
    num_features: i32,
    depth: usize,
) -> Result<ForcedSplitNode, ForcedSplitError> {
    if depth > MAX_FORCED_DEPTH {
        return Err(ForcedSplitError::TooDeep);
    }
    let items = match value {
        JsonValue::Object(items) => items,
        _ => return Err(ForcedSplitError::WrongType("node")),
    };
    let feature = match object_get(items, "feature") {
        Some(JsonValue::Number(n)) => *n as i32,
        Some(_) => return Err(ForcedSplitError::WrongType("feature")),
        None => return Err(ForcedSplitError::MissingKey("feature")),
    };
    if feature < 0 || feature >= num_features {
        return Err(ForcedSplitError::FeatureOutOfRange { feature, num_features });
    }
    let threshold = match object_get(items, "threshold") {
        Some(JsonValue::Number(n)) => *n,
        Some(_) => return Err(ForcedSplitError::WrongType("threshold")),
        None => return Err(ForcedSplitError::MissingKey("threshold")),
    };
    if !threshold.is_finite() {
        return Err(ForcedSplitError::NonFiniteThreshold);
    }
    let left = match object_get(items, "left") {
        Some(v @ JsonValue::Object(_)) => {
            Some(Box::new(value_to_node(v, num_features, depth + 1)?))
        }
        _ => None,
    };
    let right = match object_get(items, "right") {
        Some(v @ JsonValue::Object(_)) => {
            Some(Box::new(value_to_node(v, num_features, depth + 1)?))
        }
        _ => None,
    };
    Ok(ForcedSplitNode {
        feature,
        threshold,
        left,
        right,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_none() {
        assert_eq!(parse_forced_splits("", 3).unwrap(), None);
        assert_eq!(parse_forced_splits("   \n", 3).unwrap(), None);
    }

    #[test]
    fn single_forced_split() {
        let json = r#"{"feature": 0, "threshold": 1.5}"#;
        let node = parse_forced_splits(json, 3).unwrap().unwrap();
        assert_eq!(node.feature, 0);
        assert_eq!(node.threshold, 1.5);
        assert!(node.left.is_none());
        assert!(node.right.is_none());
    }

    #[test]
    fn nested_left_right() {
        let json = r#"{
            "feature": 0, "threshold": 1.5,
            "left":  {"feature": 1, "threshold": 2.0},
            "right": {"feature": 2, "threshold": 3.0}
        }"#;
        let node = parse_forced_splits(json, 3).unwrap().unwrap();
        assert_eq!(node.feature, 0);
        assert_eq!(node.left.as_ref().unwrap().feature, 1);
        assert_eq!(node.right.as_ref().unwrap().feature, 2);
        assert_eq!(node.right.as_ref().unwrap().threshold, 3.0);
    }

    #[test]
    fn malformed_missing_feature_is_typed_error() {
        let json = r#"{"threshold": 1.5}"#;
        assert_eq!(
            parse_forced_splits(json, 3).unwrap_err(),
            ForcedSplitError::MissingKey("feature")
        );
    }

    #[test]
    fn malformed_syntax_is_typed_error_not_panic() {
        let json = r#"{"feature": 0, "threshold": }"#;
        assert!(matches!(
            parse_forced_splits(json, 3),
            Err(ForcedSplitError::Syntax(_))
        ));
    }

    #[test]
    fn feature_out_of_range_rejected() {
        let json = r#"{"feature": 9, "threshold": 1.5}"#;
        assert_eq!(
            parse_forced_splits(json, 3).unwrap_err(),
            ForcedSplitError::FeatureOutOfRange {
                feature: 9,
                num_features: 3
            }
        );
    }

    #[test]
    fn deeply_nested_rejected() {
        // Build a forced-split chain deeper than MAX_FORCED_DEPTH.
        let mut json = String::new();
        let depth = MAX_FORCED_DEPTH + 5;
        for _ in 0..depth {
            json.push_str(r#"{"feature":0,"threshold":1.0,"left":"#);
        }
        json.push_str(r#"{"feature":0,"threshold":1.0}"#);
        for _ in 0..depth {
            json.push('}');
        }
        assert_eq!(
            parse_forced_splits(&json, 3).unwrap_err(),
            ForcedSplitError::TooDeep
        );
    }
}
