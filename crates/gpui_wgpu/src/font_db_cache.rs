//! An on-disk cache of the parsed system font database.
//!
//! `cosmic_text::FontSystem::new()` walks every font directory and parses the name, OS/2 and post
//! tables of every face. On a machine with a few thousand faces that costs well over a second
//! when the font files are not already in the page cache, which is what a short-lived process
//! such as a lock screen sees on a cold start.
//!
//! This module stores the resulting `fontdb` face list. A cache hit rebuilds the identical
//! database from it without opening a single font file; fonts are still memory mapped on demand
//! when a face is first shaped, so fallback fonts (CJK, emoji, ...) keep working exactly as
//! before. The cache is validated against the mtime of every directory fontconfig would scan, so
//! it is rebuilt whenever a font file is added, removed or replaced through a rename. (Rewriting
//! a font file in place without touching its directory is not noticed.)
//!
//! Restoring the database does not make the fonts themselves cheap: when cosmic-text matches a
//! font it probes every face for a variable weight axis, which opens every font file. On a cold
//! page cache that serial walk is as slow as parsing was, so a restore also starts a background
//! prefetch that reads the headers those probes need from many threads at once.
//!
//! Face language tags are not stored (they are not used for matching), so every cached family
//! name is tagged `English_UnitedStates`.

use cosmic_text::fontdb::{self, Database, FaceInfo, Language, Source, Style, Weight};
use std::{
    collections::HashSet,
    ffi::OsStr,
    io::{self, Read as _, Write as _},
    os::unix::ffi::OsStrExt as _,
    path::{Path, PathBuf},
    time::{Instant, UNIX_EPOCH},
};

const MAGIC: &[u8; 8] = b"GPUIFDB\x03";

/// Environment variable that opts into the cache. `1`/`auto` uses the default location under the
/// user cache directory, any other non-empty value is used as the cache file path.
pub const CACHE_ENV: &str = "GPUI_FONT_DB_CACHE";

pub fn cache_path_from_env() -> Option<PathBuf> {
    let value = std::env::var_os(CACHE_ENV)?;
    if value.is_empty() || value == "0" {
        return None;
    }
    if value == "1" || value == "auto" {
        return default_cache_path();
    }
    Some(PathBuf::from(value))
}

pub fn default_cache_path() -> Option<PathBuf> {
    let base = match std::env::var_os("XDG_CACHE_HOME") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => PathBuf::from(std::env::var_os("HOME")?).join(".cache"),
    };
    Some(base.join("gpui").join("fontdb-v3.bin"))
}

/// Loads the system fonts into `db`, going through the cache at `path`.
pub fn load_system_fonts(db: &mut Database, path: &Path) {
    let started = Instant::now();
    let roots = font_roots();

    if let Some(cache) = read_cache(path, &roots) {
        let mut prefetch = Vec::with_capacity(cache.faces.len());
        for face in cache.faces {
            if let Source::File(file) = &face.source {
                prefetch.push((file.clone(), face.index));
            }
            db.push_face_info(face);
        }
        spawn_prefetch(prefetch);
        log::debug!(
            "Restored {} font faces from {} in {}ms.",
            db.len(),
            path.display(),
            started.elapsed().as_millis()
        );
        return;
    }

    db.load_system_fonts();
    log::debug!(
        "Parsed {} font faces in {}ms.",
        db.len(),
        started.elapsed().as_millis()
    );

    match write_cache(path, &roots, db) {
        Ok(()) => log::debug!("Wrote font cache to {}.", path.display()),
        Err(error) => log::debug!("Could not write font cache {}: {error}", path.display()),
    }
}

/// The directories `fontdb::Database::load_system_fonts` scans on Linux, mirroring its logic.
fn font_roots() -> Vec<PathBuf> {
    let mut config = fontconfig_parser::FontConfig::default();
    let home = std::env::var_os("HOME").map(PathBuf::from);

    if let Some(file) = std::env::var_os("FONTCONFIG_FILE") {
        let _ = config.merge_config(Path::new(&file));
    } else {
        let xdg_config_home = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| home.as_ref().map(|home| home.join(".config")));
        let read_global = match xdg_config_home {
            Some(dir) => config
                .merge_config(&dir.join("fontconfig/fonts.conf"))
                .is_err(),
            None => true,
        };
        if read_global {
            let _ = config.merge_config(Path::new("/etc/fonts/local.conf"));
        }
        let _ = config.merge_config(Path::new("/etc/fonts/fonts.conf"));
    }

    let mut roots = Vec::new();
    for dir in config.dirs {
        let path = match dir.path.strip_prefix("~") {
            Ok(rest) => match &home {
                Some(home) => home.join(rest),
                None => continue,
            },
            Err(_) => dir.path,
        };
        roots.push(path);
    }

    if roots.is_empty() {
        roots.push("/usr/share/fonts/".into());
        roots.push("/usr/local/share/fonts/".into());
        if let Some(home) = &home {
            roots.push(home.join(".fonts"));
            roots.push(home.join(".local/share/fonts"));
        }
    }
    roots
}

/// Modification time in nanoseconds since the epoch, `0` when the path does not exist.
fn stamp(path: &Path) -> u128 {
    std::fs::metadata(path).map_or(0, |meta| {
        meta.modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map_or(1, |duration| duration.as_nanos().max(1))
    })
}

/// Every directory below `roots` (symlinks followed, cycles skipped), roots included.
fn walk_dirs(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let mut seen = HashSet::new();
    let mut pending: Vec<PathBuf> = roots.iter().rev().cloned().collect();
    while let Some(dir) = pending.pop() {
        let Ok(canonical) = std::fs::canonicalize(&dir) else {
            dirs.push(dir);
            continue;
        };
        if !seen.insert(canonical) {
            continue;
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if std::fs::metadata(&path).is_ok_and(|meta| meta.is_dir()) {
                    pending.push(path);
                }
            }
        }
        dirs.push(dir);
    }
    dirs
}

struct Cache {
    faces: Vec<FaceInfo>,
}

fn write_cache(path: &Path, roots: &[PathBuf], db: &Database) -> io::Result<()> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);

    put_u32(&mut out, roots.len() as u32);
    for root in roots {
        put_path(&mut out, root);
    }

    let dirs = walk_dirs(roots);
    put_u32(&mut out, dirs.len() as u32);
    for dir in &dirs {
        put_path(&mut out, dir);
        put_stamp(&mut out, stamp(dir));
    }

    // Faces from one collection share a file; list each file once.
    let mut files: Vec<&Path> = Vec::new();
    let mut file_ids = std::collections::HashMap::new();
    let mut faces = Vec::new();
    for face in db.faces() {
        let (Source::File(file) | Source::SharedFile(file, _)) = &face.source else {
            // Embedded fonts are not part of the system set.
            continue;
        };
        let file_id = *file_ids.entry(file.as_path()).or_insert_with(|| {
            files.push(file.as_path());
            files.len() - 1
        });
        faces.push((file_id, face));
    }

    put_u32(&mut out, files.len() as u32);
    for file in &files {
        put_path(&mut out, file);
    }

    put_u32(&mut out, faces.len() as u32);
    for (file_id, face) in faces {
        put_u32(&mut out, file_id as u32);
        put_u32(&mut out, face.index);
        put_u32(&mut out, face.families.len() as u32);
        for (name, _) in &face.families {
            put_str(&mut out, name);
        }
        put_str(&mut out, &face.post_script_name);
        out.push(match face.style {
            Style::Normal => 0,
            Style::Italic => 1,
            Style::Oblique => 2,
        });
        out.extend_from_slice(&face.weight.0.to_le_bytes());
        out.push(stretch_to_u8(face.stretch));
        out.push(face.monospaced as u8);
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Several processes may start at once; write privately and rename into place.
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    let result = std::fs::File::create(&tmp)
        .and_then(|mut file| file.write_all(&out))
        .and_then(|()| std::fs::rename(&tmp, path));
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

fn read_cache(path: &Path, roots: &[PathBuf]) -> Option<Cache> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .ok()?
        .read_to_end(&mut bytes)
        .ok()?;
    let mut input = Reader(&bytes);

    if input.take(MAGIC.len())? != MAGIC {
        return None;
    }

    let cached_roots = input.u32()? as usize;
    if cached_roots != roots.len() {
        return None;
    }
    for root in roots {
        if input.path()? != root.as_path() {
            return None;
        }
    }

    for _ in 0..input.u32()? {
        let dir = input.path()?;
        if input.stamp()? != stamp(dir) {
            return None;
        }
    }

    let mut files = Vec::new();
    for _ in 0..input.u32()? {
        files.push(input.path()?.to_path_buf());
    }

    let mut faces = Vec::new();
    for _ in 0..input.u32()? {
        let file = files.get(input.u32()? as usize)?.clone();
        let index = input.u32()?;
        let mut families = Vec::new();
        for _ in 0..input.u32()? {
            families.push((input.str()?.to_owned(), Language::English_UnitedStates));
        }
        let post_script_name = input.str()?.to_owned();
        let style = match input.u8()? {
            0 => Style::Normal,
            1 => Style::Italic,
            2 => Style::Oblique,
            _ => return None,
        };
        let weight = Weight(input.u16()?);
        let stretch = stretch_from_u8(input.u8()?)?;
        let monospaced = input.u8()? != 0;
        faces.push(FaceInfo {
            id: fontdb::ID::dummy(),
            source: Source::File(file),
            index,
            families,
            post_script_name,
            style,
            weight,
            stretch,
            monospaced,
        });
    }

    Some(Cache { faces })
}

fn stretch_to_u8(stretch: fontdb::Stretch) -> u8 {
    use fontdb::Stretch::*;
    match stretch {
        UltraCondensed => 0,
        ExtraCondensed => 1,
        Condensed => 2,
        SemiCondensed => 3,
        Normal => 4,
        SemiExpanded => 5,
        Expanded => 6,
        ExtraExpanded => 7,
        UltraExpanded => 8,
    }
}

fn stretch_from_u8(value: u8) -> Option<fontdb::Stretch> {
    use fontdb::Stretch::*;
    Some(match value {
        0 => UltraCondensed,
        1 => ExtraCondensed,
        2 => Condensed,
        3 => SemiCondensed,
        4 => Normal,
        5 => SemiExpanded,
        6 => Expanded,
        7 => ExtraExpanded,
        8 => UltraExpanded,
        _ => return None,
    })
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_u32(out, bytes.len() as u32);
    out.extend_from_slice(bytes);
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_bytes(out, value.as_bytes());
}

fn put_path(out: &mut Vec<u8>, path: &Path) {
    put_bytes(out, path.as_os_str().as_bytes());
}

fn put_stamp(out: &mut Vec<u8>, stamp: u128) {
    out.extend_from_slice(&stamp.to_le_bytes());
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let (head, tail) = self.0.split_at_checked(len)?;
        self.0 = tail;
        Some(head)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn str(&mut self) -> Option<&'a str> {
        let len = self.u32()? as usize;
        std::str::from_utf8(self.take(len)?).ok()
    }

    fn path(&mut self) -> Option<&'a Path> {
        let len = self.u32()? as usize;
        Some(Path::new(OsStr::from_bytes(self.take(len)?)))
    }

    fn stamp(&mut self) -> Option<u128> {
        Some(u128::from_le_bytes(self.take(16)?.try_into().ok()?))
    }
}


/// Warms the page cache with the parts of each font file that matching a font reads: the table
/// directory and the `fvar` table. Runs on detached background threads.
fn spawn_prefetch(faces: Vec<(PathBuf, u32)>) {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    let faces = Arc::new(faces);
    let next = Arc::new(AtomicUsize::new(0));
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().clamp(4, 16));
    for _ in 0..threads {
        let faces = faces.clone();
        let next = next.clone();
        let spawned = std::thread::Builder::new()
            .name("font-prefetch".into())
            .spawn(move || {
                loop {
                    let ix = next.fetch_add(1, Ordering::Relaxed);
                    let Some((path, index)) = faces.get(ix) else {
                        break;
                    };
                    let _ = prefetch_face(path, *index);
                }
            });
        if spawned.is_err() {
            break;
        }
    }
}

fn prefetch_face(path: &Path, index: u32) -> io::Result<()> {
    use std::os::unix::fs::FileExt as _;

    let file = std::fs::File::open(path)?;
    let mut head = [0u8; 4096];
    let read = file.read_at(&mut head, 0)?;
    let head = &head[..read];

    // A collection starts with a header listing the offset of each face's table directory.
    let directory = if head.get(..4) == Some(b"ttcf") {
        let entry = 12 + 4 * index as usize;
        match head.get(entry..entry + 4) {
            Some(offset) => u32::from_be_bytes(offset.try_into().unwrap()) as u64,
            None => return Ok(()),
        }
    } else {
        0
    };

    let mut header = [0u8; 12];
    file.read_exact_at(&mut header, directory)?;
    let table_count = u16::from_be_bytes([header[4], header[5]]) as usize;
    let mut records = vec![0u8; table_count * 16];
    file.read_exact_at(&mut records, directory + 12)?;

    for record in records.chunks_exact(16) {
        if &record[..4] == b"fvar" {
            let offset = u32::from_be_bytes(record[8..12].try_into().unwrap()) as u64;
            let length = u32::from_be_bytes(record[12..16].try_into().unwrap()) as usize;
            let mut fvar = vec![0u8; length.min(64 * 1024)];
            file.read_exact_at(&mut fvar, offset)?;
            break;
        }
    }
    Ok(())
}
