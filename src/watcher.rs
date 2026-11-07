//! Folder watching: finds audio files already present and new arrivals,
//! and yields each one once, sequentially, when it is fully written.

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::thread::sleep;
use std::time::Duration;

use anyhow::{Context, Result};
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};

/// Extensions we accept (lowercase). QuickTime and Voice Memos produce `m4a`.
const AUDIO_EXTENSIONS: &[&str] = &["m4a", "mp3", "wav", "aac", "flac"];

/// True for regular-looking, non-hidden audio files.
pub fn is_audio_candidate(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    if name.starts_with('.') || name.starts_with("~$") {
        return false;
    }
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| AUDIO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

/// Audio files already in `dir` (non-recursive), sorted by name.
pub fn scan_existing(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && is_audio_candidate(p))
        .collect();
    files.sort();
    Ok(files)
}

/// Blocks until the file size is unchanged for `required_stable` consecutive
/// checks spaced by `interval`. Returns false if the file disappears.
pub fn wait_until_stable(path: &Path, interval: Duration, required_stable: u32) -> bool {
    let mut last = None;
    let mut stable = 0;
    loop {
        let Ok(meta) = std::fs::metadata(path) else {
            return false;
        };
        let size = meta.len();
        if size > 0 && last == Some(size) {
            stable += 1;
            if stable >= required_stable {
                return true;
            }
        } else {
            stable = 0;
            last = Some(size);
        }
        sleep(interval);
    }
}

/// De-duplicating FIFO of paths fed by a channel.
pub struct FileQueue {
    rx: Receiver<PathBuf>,
    queue: VecDeque<PathBuf>,
    queued: HashSet<PathBuf>,
}

impl FileQueue {
    pub fn new(rx: Receiver<PathBuf>) -> Self {
        Self {
            rx,
            queue: VecDeque::new(),
            queued: HashSet::new(),
        }
    }

    fn push(&mut self, path: PathBuf) {
        if self.queued.insert(path.clone()) {
            self.queue.push_back(path);
        }
    }

    /// Next path to process, blocking until one arrives.
    /// Returns `None` when the sender side is gone and the queue is empty.
    pub fn next(&mut self) -> Option<PathBuf> {
        while let Ok(p) = self.rx.try_recv() {
            self.push(p);
        }
        if self.queue.is_empty() {
            let p = self.rx.recv().ok()?;
            self.push(p);
            while let Ok(p) = self.rx.try_recv() {
                self.push(p);
            }
        }
        let path = self.queue.pop_front()?;
        self.queued.remove(&path);
        Some(path)
    }
}

/// Starts watching `dir`. Existing files are queued first, then new ones.
/// The returned watcher must be kept alive for events to keep flowing.
pub fn watch(dir: &Path) -> Result<(RecommendedWatcher, FileQueue)> {
    let (tx, rx) = mpsc::channel();

    let event_tx = tx.clone();
    let mut watcher =
        notify::recommended_watcher(move |res: notify::Result<notify::Event>| match res {
            Ok(event) => {
                if matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_)) {
                    for path in event.paths {
                        if is_audio_candidate(&path) {
                            let _ = event_tx.send(path);
                        }
                    }
                }
            }
            Err(e) => tracing::warn!("watch error: {e}"),
        })
        .context("cannot create file watcher")?;
    watcher
        .watch(dir, RecursiveMode::NonRecursive)
        .with_context(|| format!("cannot watch {}", dir.display()))?;

    // Scan after the watcher is active so nothing is missed in between.
    for path in scan_existing(dir)? {
        let _ = tx.send(path);
    }

    Ok((watcher, FileQueue::new(rx)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn filters_candidates() {
        assert!(is_audio_candidate(Path::new("a/Nuova registrazione.m4a")));
        assert!(is_audio_candidate(Path::new("a/B.M4A")));
        assert!(!is_audio_candidate(Path::new("a/.hidden.m4a")));
        assert!(!is_audio_candidate(Path::new("a/.DS_Store")));
        assert!(!is_audio_candidate(Path::new("a/.gitkeep")));
        assert!(!is_audio_candidate(Path::new("a/notes.txt")));
        assert!(!is_audio_candidate(Path::new("a/song.m4a.download")));
        assert!(!is_audio_candidate(Path::new("a/noext")));
    }

    #[test]
    fn scan_finds_only_audio_sorted() {
        let d = tempfile::tempdir().unwrap();
        for n in ["b.m4a", "a.m4a", ".gitkeep", "x.txt"] {
            fs::write(d.path().join(n), b"x").unwrap();
        }
        fs::create_dir(d.path().join("sub.m4a")).unwrap();
        let found = scan_existing(d.path()).unwrap();
        let names: Vec<_> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, ["a.m4a", "b.m4a"]);
    }

    #[test]
    fn stable_detects_finished_and_missing_files() {
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("a.m4a");
        fs::write(&f, b"data").unwrap();
        assert!(wait_until_stable(&f, Duration::from_millis(10), 2));
        assert!(!wait_until_stable(
            &d.path().join("gone.m4a"),
            Duration::from_millis(10),
            2
        ));
    }

    #[test]
    fn queue_dedupes_and_preserves_order() {
        let (tx, rx) = mpsc::channel();
        for n in ["a", "b", "a", "c", "b"] {
            tx.send(PathBuf::from(n)).unwrap();
        }
        drop(tx);
        let mut q = FileQueue::new(rx);
        let got: Vec<_> = std::iter::from_fn(|| q.next()).collect();
        assert_eq!(got, [PathBuf::from("a"), "b".into(), "c".into()]);
    }

    #[test]
    fn watch_reports_existing_and_new_files() {
        let d = tempfile::tempdir().unwrap();
        fs::write(d.path().join("old.m4a"), b"x").unwrap();
        let (_w, mut q) = watch(d.path()).unwrap();
        assert_eq!(q.next().unwrap().file_name().unwrap(), "old.m4a");
        fs::write(d.path().join("new.m4a"), b"x").unwrap();
        assert_eq!(q.next().unwrap().file_name().unwrap(), "new.m4a");
    }
}
