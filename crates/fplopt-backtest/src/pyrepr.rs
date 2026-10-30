//! Python reprs for error messages that name dicts and lists, as Python prints them.

/// `repr(dict)` of int pairs, in the given order: `{1: 2, 3: 5}`.
pub fn dict<'a>(pairs: impl IntoIterator<Item = (&'a i64, &'a i64)>) -> String {
    let body: Vec<String> = pairs.into_iter().map(|(k, v)| format!("{k}: {v}")).collect();
    format!("{{{}}}", body.join(", "))
}

/// `repr(list)` of ints: `[1, 2]`.
pub fn list(items: &[i64]) -> String {
    let body: Vec<String> = items.iter().map(i64::to_string).collect();
    format!("[{}]", body.join(", "))
}

/// A `collections.Counter` in first-seen order (what `dict(Counter(...))` prints).
pub fn counter(items: impl IntoIterator<Item = i64>) -> Vec<(i64, i64)> {
    let mut out: Vec<(i64, i64)> = Vec::new();
    for item in items {
        match out.iter_mut().find(|(k, _)| *k == item) {
            Some(entry) => entry.1 += 1,
            None => out.push((item, 1)),
        }
    }
    out
}

/// `repr(dict)` of a first-seen-order counter.
pub fn counter_dict(counts: &[(i64, i64)]) -> String {
    dict(counts.iter().map(|(k, v)| (k, v)))
}
