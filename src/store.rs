//! Where haico keeps its own files.
//!
//! One directory per machine, holding the config file, the user's instructions, the saved
//! sessions, the input history and the debug log. It used to be `~/.haico`, with the input
//! history not even in it (`~/.haico_history`, a second dotfile in the home directory).
//! That is the convention of a tool that predates the platform having an opinion, and on
//! Windows it is simply wrong: a home directory is not where an application's data goes,
//! and a roaming profile will not carry it.
//!
//! So the location is the platform's now:
//!
//! | | |
//! | --- | --- |
//! | Windows | `%LOCALAPPDATA%\Haico` |
//! | macOS | `~/Library/Application Support/Haico` |
//! | everything else | `$XDG_DATA_HOME/haico`, or `~/.local/share/haico` |
//!
//! **One directory rather than the XDG split**, even on Linux where the spec offers
//! config, data, state and cache separately. Windows and macOS each give an application
//! one place to put things, so honoring the split would mean two layouts to reason about
//! and two answers to "where is my config" depending on the machine. The single directory
//! is the shape all three platforms share.
//!
//! **There is no migration from `~/.haico`, deliberately.** Reading the old location as a
//! fallback would mean carrying two layouts indefinitely, and moving files on the user's
//! behalf at startup is the kind of thing that is fine until the one time it is not.
//! Whoever has an old directory moves it themselves, once.

use std::env;
use std::path::PathBuf;

/// The directory haico stores everything of its own in.
///
/// `None` only when the environment names no home at all, which is a machine haico cannot
/// save anything on. Every caller already handles that — the config file, the session and
/// the history are each optional in their own right.
pub(crate) fn dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        windows_dir(var("LOCALAPPDATA"), var("USERPROFILE"))
    }
    #[cfg(target_os = "macos")]
    {
        macos_dir(var("HOME"))
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        xdg_dir(var("XDG_DATA_HOME"), var("HOME"))
    }
}

/// One of haico's own files, by name.
pub(crate) fn path(name: &str) -> Option<PathBuf> {
    Some(dir()?.join(name))
}

/// The same path, for help text and for the system prompt.
///
/// The resolved absolute path rather than a `%LOCALAPPDATA%`-style placeholder: the reader
/// is either a person about to open the file or the model about to edit it, and neither is
/// helped by being handed something they have to expand first.
pub(crate) fn display(name: &str) -> String {
    match path(name) {
        Some(path) => crate::util::display_path(&path),
        None => format!("<haico data directory>/{}", name),
    }
}

fn var(name: &str) -> Option<PathBuf> {
    env::var_os(name).map(PathBuf::from)
}

// The three resolvers are pure and compiled on every platform, with only the caller above
// behind a `cfg`. Rust 2024 makes setting an environment variable unsafe — rightly, since
// the tests run in parallel in one process and a second thread reading while one writes is
// a data race — so a resolver that read the environment itself could only ever be tested
// on the platform it is for, which is to say not at all from here.

#[allow(dead_code)] // dead on two of the three targets; see above.
fn windows_dir(local_appdata: Option<PathBuf>, user_profile: Option<PathBuf>) -> Option<PathBuf> {
    // LOCALAPPDATA first, and not merely as a shortcut: a redirected or roaming profile
    // puts it somewhere other than under USERPROFILE, and the hand-joined path would then
    // name a directory Windows itself is not using.
    let base = usable(local_appdata)
        .or_else(|| usable(user_profile).map(|home| home.join("AppData").join("Local")))?;
    Some(base.join("Haico"))
}

#[allow(dead_code)] // dead on two of the three targets; see above.
fn macos_dir(home: Option<PathBuf>) -> Option<PathBuf> {
    Some(usable(home)?.join("Library").join("Application Support").join("Haico"))
}

#[allow(dead_code)] // dead on two of the three targets; see above.
fn xdg_dir(xdg_data_home: Option<PathBuf>, home: Option<PathBuf>) -> Option<PathBuf> {
    // The spec is explicit that a relative path in one of these variables is invalid and
    // must be ignored rather than resolved against the working directory — which for haico
    // would mean writing the user's config into whichever project it was started in.
    //
    // A leading slash rather than `Path::is_absolute`, which answers by the rules of
    // whichever platform it was compiled for: on Windows it calls "/home/someone" relative,
    // so the check would depend on the host rather than on the value, and the test for it
    // could only pass on Unix. This is a Unix spec and a leading slash is what it means.
    let base = usable(xdg_data_home)
        .filter(|path| path.to_string_lossy().starts_with('/'))
        .or_else(|| usable(home).map(|home| home.join(".local").join("share")))?;
    Some(base.join("haico"))
}

// An environment variable that is set but empty is not set. Windows and the XDG spec agree
// on this, and `env::var_os` does not do it for us.
fn usable(path: Option<PathBuf>) -> Option<PathBuf> {
    path.filter(|path| !path.as_os_str().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Option<PathBuf> {
        Some(PathBuf::from(s))
    }

    #[test]
    fn windows_prefers_local_appdata_over_a_hand_built_profile_path() {
        assert_eq!(
            windows_dir(p(r"D:\redirected\Local"), p(r"C:\Users\someone")).unwrap(),
            PathBuf::from(r"D:\redirected\Local\Haico")
        );
    }

    #[test]
    fn windows_falls_back_to_the_profile_when_local_appdata_is_absent_or_empty() {
        let expected = PathBuf::from(r"C:\Users\someone\AppData\Local\Haico");
        assert_eq!(windows_dir(None, p(r"C:\Users\someone")).unwrap(), expected);
        assert_eq!(windows_dir(p(""), p(r"C:\Users\someone")).unwrap(), expected);
    }

    #[test]
    fn windows_has_nowhere_to_go_without_either() {
        assert!(windows_dir(None, None).is_none());
        assert!(windows_dir(p(""), p("")).is_none());
    }

    #[test]
    fn macos_uses_application_support() {
        assert_eq!(
            macos_dir(p("/Users/someone")).unwrap(),
            PathBuf::from("/Users/someone/Library/Application Support/Haico")
        );
        assert!(macos_dir(None).is_none());
    }

    #[test]
    fn xdg_data_home_wins_when_it_is_set() {
        assert_eq!(
            xdg_dir(p("/home/someone/.data"), p("/home/someone")).unwrap(),
            PathBuf::from("/home/someone/.data/haico")
        );
    }

    #[test]
    fn xdg_falls_back_to_the_spec_s_default() {
        let expected = PathBuf::from("/home/someone/.local/share/haico");
        assert_eq!(xdg_dir(None, p("/home/someone")).unwrap(), expected);
        assert_eq!(xdg_dir(p(""), p("/home/someone")).unwrap(), expected);
    }

    #[test]
    fn a_relative_xdg_data_home_is_ignored_rather_than_resolved() {
        // Resolving it against the working directory would write the user's config into
        // whichever project haico happened to be started in.
        assert_eq!(
            xdg_dir(p("relative/path"), p("/home/someone")).unwrap(),
            PathBuf::from("/home/someone/.local/share/haico")
        );
    }

    #[test]
    fn xdg_has_nowhere_to_go_without_either() {
        assert!(xdg_dir(None, None).is_none());
    }

    #[test]
    fn the_platform_directory_ends_in_the_application_s_name() {
        // Whichever platform this is running on, `dir()` has to agree with its resolver.
        let dir = dir().expect("this machine names a home directory");
        let last = dir.file_name().unwrap().to_string_lossy().to_string();
        assert!(
            last == "Haico" || last == "haico",
            "the store landed somewhere unexpected: {}",
            dir.display()
        );
    }
}
