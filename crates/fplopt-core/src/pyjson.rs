//! JSON written the way Python's `json.dumps` writes it (", " and ": " separators,
//! `ensure_ascii`), so files and log rows from both implementations match byte for byte.
//! Field order is the value's own: a struct serializes in declaration order (like a Python
//! dict literal), a `serde_json::Value` with sorted keys (like `sort_keys=True`).

use std::io;

use anyhow::Result;
use serde::Serialize;
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq, Serializer};
use serde_json::ser::{CharEscape, Formatter};

/// `json.dumps(value)`.
pub fn dumps<T: Serialize + ?Sized>(value: &T) -> Result<String> {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PythonFormatter);
    value.serialize(&mut ser)?;
    Ok(String::from_utf8(out)?)
}

/// Python's `repr(float)`: shortest round-trip digits, scientific below 1e-4 or from 1e16.
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let sci = format!("{f:e}"); // shortest digits, e.g. "1.5e-5"
    let (mantissa, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let sign = if negative { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let point = exp + 1; // digits before the decimal point
        let body = if point <= 0 {
            format!("0.{}{digits}", "0".repeat((-point) as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
        };
        format!("{sign}{body}")
    } else {
        let rest = if digits.len() > 1 { format!(".{}", &digits[1..]) } else { String::new() };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{}{rest}e{esign}{:02}", &digits[..1], exp.abs())
    }
}

/// A JSON value as Python's `json.loads` holds it: objects keep their key order (a repeated
/// key keeps its first place and its last value), integers stay integers. `dumps` of it is
/// `json.dumps(json.loads(text))`.
#[derive(Clone, Debug, PartialEq)]
pub enum Ordered {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    Array(Vec<Ordered>),
    Object(Vec<(String, Ordered)>),
}

impl Ordered {
    pub fn parse(bytes: &[u8]) -> Result<Ordered> {
        Ok(serde_json::from_slice(bytes)?)
    }
}

impl Serialize for Ordered {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Ordered::Null => s.serialize_unit(),
            Ordered::Bool(b) => s.serialize_bool(*b),
            Ordered::Int(i) => s.serialize_i128(*i),
            Ordered::Float(f) => s.serialize_f64(*f),
            Ordered::Str(x) => s.serialize_str(x),
            Ordered::Array(items) => {
                let mut seq = s.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Ordered::Object(pairs) => {
                let mut map = s.serialize_map(Some(pairs.len()))?;
                for (k, v) in pairs {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Ordered {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Ordered, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Ordered;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_unit<E: de::Error>(self) -> Result<Ordered, E> {
                Ok(Ordered::Null)
            }
            fn visit_bool<E: de::Error>(self, b: bool) -> Result<Ordered, E> {
                Ok(Ordered::Bool(b))
            }
            fn visit_i64<E: de::Error>(self, i: i64) -> Result<Ordered, E> {
                Ok(Ordered::Int(i.into()))
            }
            fn visit_u64<E: de::Error>(self, i: u64) -> Result<Ordered, E> {
                Ok(Ordered::Int(i.into()))
            }
            fn visit_f64<E: de::Error>(self, f: f64) -> Result<Ordered, E> {
                Ok(Ordered::Float(f))
            }
            fn visit_str<E: de::Error>(self, x: &str) -> Result<Ordered, E> {
                Ok(Ordered::Str(x.to_owned()))
            }
            fn visit_string<E: de::Error>(self, x: String) -> Result<Ordered, E> {
                Ok(Ordered::Str(x))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Ordered, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(Ordered::Array(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Ordered, A::Error> {
                let mut pairs: Vec<(String, Ordered)> = Vec::new();
                while let Some((k, v)) = map.next_entry::<String, Ordered>()? {
                    match pairs.iter_mut().find(|(key, _)| *key == k) {
                        Some(pair) => pair.1 = v,
                        None => pairs.push((k, v)),
                    }
                }
                Ok(Ordered::Object(pairs))
            }
        }
        d.deserialize_any(V)
    }
}

struct PythonFormatter;

impl Formatter for PythonFormatter {
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b": ")
    }

    /// `ensure_ascii`: anything outside ASCII as `\uXXXX` (UTF-16 surrogate pairs above
    /// U+FFFF).
    fn write_string_fragment<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        for c in fragment.chars() {
            if c.is_ascii() {
                w.write_all(&[c as u8])?;
            } else {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    write!(w, "\\u{unit:04x}")?;
                }
            }
        }
        Ok(())
    }

    /// Python's float repr (`1e-05`, `1e+16`, `2.0`); non-finite floats never get here
    /// (serde_json writes null).
    fn write_f64<W: ?Sized + io::Write>(&mut self, w: &mut W, value: f64) -> io::Result<()> {
        w.write_all(py_float_repr(value).as_bytes())
    }

    fn write_char_escape<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        escape: CharEscape,
    ) -> io::Result<()> {
        // Python writes \u00XX for control characters other than \b \f \n \r \t, as
        // serde_json does; only the hex case differs (serde_json writes lowercase too).
        let s: &[u8] = match escape {
            CharEscape::Quote => b"\\\"",
            CharEscape::ReverseSolidus => b"\\\\",
            CharEscape::Backspace => b"\\b",
            CharEscape::FormFeed => b"\\f",
            CharEscape::LineFeed => b"\\n",
            CharEscape::CarriageReturn => b"\\r",
            CharEscape::Tab => b"\\t",
            CharEscape::AsciiControl(byte) => return write!(w, "\\u{byte:04x}"),
            _ => return Err(io::Error::other("unknown JSON escape")),
        };
        w.write_all(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn separators_and_sorted_value_keys() {
        let value = json!({"b": [1, 2.5, null], "a": {"y": "x", "x": true}, "nan": f64::NAN});
        assert_eq!(
            dumps(&value).unwrap(),
            r#"{"a": {"x": true, "y": "x"}, "b": [1, 2.5, null], "nan": null}"#
        );
    }

    #[test]
    fn floats_as_python_writes_them() {
        assert_eq!(
            dumps(&[1e-5, 2.0, 1e16, 0.1, -1.25e-7]).unwrap(),
            "[1e-05, 2.0, 1e+16, 0.1, -1.25e-07]"
        );
    }

    #[test]
    fn structs_keep_field_order() {
        #[derive(Serialize)]
        struct M {
            z: i32,
            a: Option<i32>,
        }
        assert_eq!(dumps(&M { z: 1, a: None }).unwrap(), r#"{"z": 1, "a": null}"#);
    }

    #[test]
    fn ensure_ascii() {
        const B: char = 0x5c as char;
        // json.dumps("Ødegaard é \U0001F600 \x01 \"q\"")
        assert_eq!(
            dumps("Ødegaard é \u{1F600} \u{1} \"q\"").unwrap(),
            // (backslashes spelled out: the expected text is the escapes themselves)
            format!("\"{B}u00d8degaard {B}u00e9 {B}ud83d{B}ude00 {B}u0001 {B}\"q{B}\"\"")
        );
    }

    #[test]
    fn ordered_round_trips_as_python() {
        // json.dumps(json.loads(text))
        let text =
            br#"{"z":1,"a":[1.50,2e2,-0.0,{"k":"x"}],"z2":null,"a":true,"big":12345678901234}"#;
        assert_eq!(
            dumps(&Ordered::parse(text).unwrap()).unwrap(),
            r#"{"z": 1, "a": true, "z2": null, "big": 12345678901234}"#
        );
        assert_eq!(
            dumps(&Ordered::parse(br#"[1.50, 2e2, -0.0, {"k": "x"}]"#).unwrap()).unwrap(),
            r#"[1.5, 200.0, -0.0, {"k": "x"}]"#
        );
    }
}
