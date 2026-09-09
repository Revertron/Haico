//! The session's own scratch directory.
//!
//! Asked to write a throwaway script or dump intermediate output, a model invents a
//! path — `C:\Temp\build.py` on Windows, `/tmp/x` on a machine whose temp is somewhere
//! else — and the write is refused by the workspace, or lands in a directory nobody
//! made and nobody cleans. It never guesses the real temp directory, because nothing
//! ever told it what that is.
//!
//! So the session makes one, names it in the system prompt, and makes it writable. The
//! directory belongs to the conversation rather than to the machine: `/clear` throws it
//! away with the history that produced its contents, and it goes on the way out.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

/// This session's scratch directory, for as long as the session lasts.
///
/// `None` inside when the directory could not be made at all (a temp folder that is
/// read-only, or full). That is not worth failing a startup over — the prompt says
/// there is no scratch directory instead, and the workspace gains no extra root.
pub(crate) struct Scratch {
    path: Option<PathBuf>,
}

impl Scratch {
    pub(crate) fn new() -> Self {
        Self { path: create() }
    }

    pub(crate) fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Throw this directory away and make another one.
    ///
    /// Called by `/clear`: the files in it are working notes for a conversation that
    /// is being discarded, and leaving them would hand the next conversation output it
    /// has no way to interpret. The new directory has a new name, which is why the
    /// system prompt is recomposed alongside it.
    pub(crate) fn renew(&mut self) {
        self.discard();
        self.path = create();
    }

    fn discard(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = fs::remove_dir_all(path);
        }
    }
}

/// Best effort, and only on the way out through the normal exit path — a kill leaves
/// the directory behind for the OS to reap, which is what the OS temp folder is for.
impl Drop for Scratch {
    fn drop(&mut self) {
        self.discard();
    }
}

fn create() -> Option<PathBuf> {
    let base = std::env::temp_dir();
    // `create_dir` rather than `create_dir_all`, so an existing directory is a failure
    // and not a silent share with whoever made it — the retry then picks another name.
    for _ in 0..8 {
        let dir = base.join(format!("haico-{:08x}", token()));
        if fs::create_dir(&dir).is_ok() {
            return Some(dir);
        }
    }
    None
}

// A name nothing else is going to pick, from the process id, the clock in nanoseconds
// and the address of a stack local. Not a security property and not trying to be one:
// it only has to stop two haico sessions started in the same second from sharing a
// directory, which the process id alone would not (a pid is reused).
fn token() -> u32 {
    let mut hasher = DefaultHasher::new();
    std::process::id().hash(&mut hasher);
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
        .hash(&mut hasher);
    let local = 0u8;
    (&local as *const u8 as usize).hash(&mut hasher);
    hasher.finish() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_session_gets_its_own_directory_and_takes_it_away_again() {
        let path = {
            let scratch = Scratch::new();
            let path = scratch.path().expect("temp dir is writable").to_path_buf();
            assert!(path.is_dir());
            assert!(path.starts_with(std::env::temp_dir()), "not under temp: {}", path.display());
            // Writable, or it is no use as a scratch directory.
            fs::write(path.join("note.txt"), "x").unwrap();
            path
        };
        // Dropped with the session, contents and all.
        assert!(!path.exists(), "left behind: {}", path.display());
    }

    #[test]
    fn renewing_moves_to_a_fresh_directory_and_removes_the_old_one() {
        let mut scratch = Scratch::new();
        let first = scratch.path().unwrap().to_path_buf();
        fs::write(first.join("half-written.py"), "print(1)").unwrap();

        scratch.renew();
        let second = scratch.path().unwrap().to_path_buf();

        assert_ne!(first, second, "a cleared conversation should not reuse the directory");
        assert!(!first.exists(), "the old conversation's files survived: {}", first.display());
        assert!(second.is_dir());
        assert_eq!(fs::read_dir(&second).unwrap().count(), 0);
    }
}
