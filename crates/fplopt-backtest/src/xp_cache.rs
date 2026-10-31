//! xP frames on disk, shared across runs (no Python counterpart): a backtest or an eval
//! refits v1 at every deadline, and the chip study's arms and the research loop's runs refit
//! the same ones. The cache lives in `<root>/<key>/<model>/<deadline>.parquet`, where the key
//! hashes every `data/` table's name, size and modification time and the running binary's
//! bytes: a rebuilt `data/` or a rebuilt binary starts a fresh cache (a needless miss at
//! worst, never a stale frame). A frame reads back as it was written: Parquet keeps the
//! floats bit for bit and NaN apart from null, and each column's variant is restored (a
//! nullable column without nulls reads back as the non-nullable one).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result, bail};
use fplopt_build::table::{Column, Table, read_parquet, write_parquet};
use fplopt_store::Micros;
use sha2::{Digest, Sha256};

/// Where `backtest run|compare` and `models eval` keep it unless told otherwise.
pub const DEFAULT_ROOT: &str = "results/.xp-cache";

pub struct DiskCache {
    dir: PathBuf,
}

/// The variant's name (the sidecar's record of it).
fn variant(c: &Column) -> &'static str {
    match c {
        Column::I64(_) => "I64",
        Column::OptI64(_) => "OptI64",
        Column::F64(_) => "F64",
        Column::OptF64(_) => "OptF64",
        Column::Str(_) => "Str",
        Column::OptStr(_) => "OptStr",
        Column::Bool(_) => "Bool",
        Column::OptBool(_) => "OptBool",
        Column::Ts(_) => "Ts",
        Column::OptTs(_) => "OptTs",
        Column::OptDate(_) => "OptDate",
    }
}

/// `c` as the variant named `want` (only the nullable widening a Parquet round trip needs).
fn restore(c: Column, want: &str) -> Result<Column> {
    fn some<T>(v: Vec<T>) -> Vec<Option<T>> {
        v.into_iter().map(Some).collect()
    }
    Ok(match (c, want) {
        (c, w) if variant(&c) == w => c,
        (Column::I64(v), "OptI64") => Column::OptI64(some(v)),
        (Column::F64(v), "OptF64") => Column::OptF64(some(v)),
        (Column::Str(v), "OptStr") => Column::OptStr(some(v)),
        (Column::Bool(v), "OptBool") => Column::OptBool(some(v)),
        (Column::Ts(v), "OptTs") => Column::OptTs(some(v)),
        (c, w) => bail!("cached column is {}, expected {w}", variant(&c)),
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// The hash of `data/`'s tables (name, size, modification time) and of `binary`'s bytes.
pub fn cache_key(data_dir: &Path, binary: &Path) -> Result<String> {
    let mut files: Vec<(String, u64, u128)> = Vec::new();
    for entry in
        std::fs::read_dir(data_dir).with_context(|| format!("read {}", data_dir.display()))?
    {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".parquet") {
            continue;
        }
        let meta = entry.metadata()?;
        let modified = meta.modified()?.duration_since(UNIX_EPOCH)?.as_nanos();
        files.push((name, meta.len(), modified));
    }
    files.sort();
    let mut digest = Sha256::new();
    for (name, len, modified) in &files {
        digest.update(format!("{name}\t{len}\t{modified}\n").as_bytes());
    }
    digest.update(b"binary\n");
    digest.update(std::fs::read(binary).with_context(|| format!("read {}", binary.display()))?);
    Ok(hex(&digest.finalize())[..16].to_owned())
}

impl DiskCache {
    /// The cache under `root` for this `data/` and the running binary.
    pub fn open(root: &Path, data_dir: &Path) -> Result<DiskCache> {
        let key = cache_key(data_dir, &std::env::current_exe()?)?;
        Ok(DiskCache { dir: root.join(key) })
    }

    /// A cache in `dir` itself (tests).
    pub fn at(dir: PathBuf) -> DiskCache {
        DiskCache { dir }
    }

    fn path(&self, model: &str, deadline: Micros) -> PathBuf {
        self.dir.join(model).join(format!("{deadline}.parquet"))
    }

    /// The frame stored for (model, deadline), if any.
    pub fn load(&self, model: &str, deadline: Micros) -> Result<Option<Table>> {
        let path = self.path(model, deadline);
        let kinds = path.with_extension("kinds");
        if !path.exists() || !kinds.exists() {
            return Ok(None);
        }
        let wanted: Vec<String> =
            std::fs::read_to_string(&kinds)?.lines().map(str::to_owned).collect();
        let table = read_parquet(&path, None)?;
        if table.columns.len() != wanted.len() {
            bail!(
                "{}: {} columns, the sidecar lists {}",
                path.display(),
                table.columns.len(),
                wanted.len()
            );
        }
        let columns = table
            .columns
            .into_iter()
            .zip(&wanted)
            .map(|((name, c), want)| Ok((name, restore(c, want)?)))
            .collect::<Result<_>>()
            .with_context(|| format!("read {}", path.display()))?;
        Ok(Some(Table { columns }))
    }

    /// Store the frame for (model, deadline): written under a name of this process's own and
    /// renamed into place, the sidecar last (a frame without its sidecar is a miss), so
    /// processes sharing the cache never read a partial file.
    pub fn store(&self, model: &str, deadline: Micros, table: &Table) -> Result<()> {
        let path = self.path(model, deadline);
        std::fs::create_dir_all(path.parent().unwrap())?;
        let tag = format!("{}.{:?}", std::process::id(), std::thread::current().id());
        let tag: String = tag.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '.').collect();
        let tmp = path.with_extension(format!("{tag}.part.parquet"));
        write_parquet(table, &tmp)?;
        let kinds: String =
            table.columns.iter().map(|(_, c)| format!("{}\n", variant(c))).collect();
        let tmp_kinds = path.with_extension(format!("{tag}.part.kinds"));
        std::fs::write(&tmp_kinds, kinds)?;
        for (from, to) in [(&tmp, path.clone()), (&tmp_kinds, path.with_extension("kinds"))] {
            if std::fs::rename(from, &to).is_err() {
                // another process stored the same frame first: keep theirs
                let _ = std::fs::remove_file(from);
                if !to.exists() {
                    bail!("could not store {}", to.display());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fplopt_features::leakcheck::describe_difference;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("fplopt-xp-cache-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn frame() -> Table {
        Table::new()
            .with("player_key", Column::I64(vec![3, 1, 2]))
            .with("gw", Column::OptI64(vec![Some(1), Some(2), Some(3)]))
            .with("xp", Column::F64(vec![0.1 + 0.2, f64::NAN, -0.0]))
            .with("p_play", Column::OptF64(vec![Some(f64::NAN), None, Some(1e-300)]))
            .with("e_minutes", Column::OptF64(vec![Some(1.0), Some(2.5), Some(90.0)]))
            .with(
                "name",
                Column::OptStr(vec![Some("a".into()), Some("b".into()), Some("c".into())]),
            )
            .with("flag", Column::Bool(vec![true, false, true]))
            .with("deadline", Column::Ts(vec![1, 2, 3]))
    }

    #[test]
    fn a_frame_reads_back_bit_for_bit() {
        let cache = DiskCache::at(scratch("roundtrip"));
        assert!(cache.load("v1", 42).unwrap().is_none());
        let t = frame();
        cache.store("v1", 42, &t).unwrap();
        let back = cache.load("v1", 42).unwrap().unwrap();
        assert_eq!(describe_difference(&t, &back), None);
        let variants = |t: &Table| t.columns.iter().map(|(_, c)| variant(c)).collect::<Vec<_>>();
        assert_eq!(variants(&t), variants(&back));
        // -0.0 keeps its sign, 0.1 + 0.2 its last bit
        let bits = |t: &Table| t.f64("xp").unwrap().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&t), bits(&back));
        assert!(cache.load("v1", 43).unwrap().is_none());
        assert!(cache.load("rolling", 42).unwrap().is_none());
    }

    #[test]
    fn a_frame_without_its_sidecar_is_a_miss() {
        let dir = scratch("sidecar");
        let cache = DiskCache::at(dir.clone());
        cache.store("v1", 7, &frame()).unwrap();
        std::fs::remove_file(dir.join("v1").join("7.kinds")).unwrap();
        assert!(cache.load("v1", 7).unwrap().is_none());
    }

    #[test]
    fn the_key_changes_with_the_data_or_the_binary() {
        let dir = scratch("key");
        let data = dir.join("data");
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(data.join("player_gw.parquet"), b"one").unwrap();
        std::fs::write(data.join("notes.txt"), b"ignored").unwrap();
        let binary = dir.join("fplopt-rs");
        std::fs::write(&binary, b"build 1").unwrap();
        let k1 = cache_key(&data, &binary).unwrap();
        assert_eq!(k1, cache_key(&data, &binary).unwrap());
        std::fs::write(data.join("notes.txt"), b"still ignored").unwrap();
        assert_eq!(k1, cache_key(&data, &binary).unwrap());
        std::fs::write(data.join("player_gw.parquet"), b"two!").unwrap();
        let k2 = cache_key(&data, &binary).unwrap();
        assert_ne!(k1, k2);
        std::fs::write(data.join("team_rating.parquet"), b"x").unwrap();
        let k3 = cache_key(&data, &binary).unwrap();
        assert_ne!(k2, k3);
        std::fs::write(&binary, b"build 2").unwrap();
        assert_ne!(k3, cache_key(&data, &binary).unwrap());
    }
}
