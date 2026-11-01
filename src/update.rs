//! `haico --update`: replace this binary with the one from the newest GitHub release.
//!
//! The release workflow (`.github/workflows/release.yml`) names each archive
//! `haico-<tag>-<target>.<zip|tar.gz>`, with the binary inside a directory of the same
//! name. This reads that layout and nothing else.

use anyhow::{bail, Context, Result};
use semver::Version;
use serde::Deserialize;
use std::io::Read;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

// `/releases/latest` skips prereleases and drafts, so a release candidate is never pushed
// onto someone who only asked for an update.
const LATEST_RELEASE: &str = "https://api.github.com/repos/Revertron/Haico/releases/latest";

// The triples the release workflow builds, which are also what its archives are named
// after. Anything else has no build to update to, and says so rather than guessing: an
// x86_64 binary unpacked onto an ARM machine replaces a haico that works with one that
// cannot start.
const TARGET: Option<&str> = if cfg!(all(windows, target_arch = "x86_64")) {
    Some("x86_64-pc-windows-msvc")
} else if cfg!(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu")) {
    Some("x86_64-unknown-linux-gnu")
} else {
    None
};
const ARCHIVE: &str = if cfg!(windows) { "zip" } else { "tar.gz" };
const BINARY: &str = if cfg!(windows) { "haico.exe" } else { "haico" };

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    html_url: String,
    assets: Vec<Asset>,
}

#[derive(Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
    size: u64,
}

// What the latest release means for this binary.
#[derive(Debug)]
enum Check<'a> {
    UpToDate,
    // A build from a commit past the last release: "updating" it would be a downgrade.
    Ahead(Version),
    Update(Version, &'a Asset),
}

impl std::fmt::Debug for Asset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

// How long the REPL waits after starting before it asks GitHub. Not at once: startup is
// already the model picker, the MCP servers and the hooks all talking at the same time,
// and an update is never urgent enough to compete with them.
const BACKGROUND_CHECK_DELAY: Duration = Duration::from_secs(20);

/// What the background check found: the text for the activity bar, once there is a
/// newer release. Shared with the task that fills it in.
#[derive(Clone, Default)]
pub(crate) struct Available {
    inner: Arc<(Mutex<Option<String>>, Notify)>,
}

impl Available {
    pub(crate) fn notice(&self) -> Option<String> {
        self.inner.0.lock().unwrap().clone()
    }

    /// Resolves once the notice has been set, so an idle prompt can redraw with it. Never
    /// resolves otherwise.
    pub(crate) async fn changed(&self) {
        self.inner.1.notified().await;
    }

    fn set(&self, notice: String) {
        *self.inner.0.lock().unwrap() = Some(notice);
        // `notify_one` keeps a permit when nobody is waiting yet, so a prompt that starts
        // waiting a moment later still wakes.
        self.inner.1.notify_one();
    }
}

/// Look for a newer release once, `BACKGROUND_CHECK_DELAY` from now, without holding
/// anything up. Every failure is silent apart from the debug log: offline, rate-limited
/// or GitHub being down are all "no news", and none of them is worth a line in someone's
/// session.
pub(crate) fn check_in_background() -> Available {
    let available = Available::default();
    // No build for this platform, so nothing it could be told to update to.
    let Some(target) = TARGET else {
        return available;
    };
    let found = available.clone();
    tokio::spawn(async move {
        tokio::time::sleep(BACKGROUND_CHECK_DELAY).await;
        let result = async {
            let release = latest_release(&client()?, LATEST_RELEASE).await?;
            Ok::<_, anyhow::Error>(match check(&release, env!("CARGO_PKG_VERSION"), target)? {
                Check::Update(latest, _) => Some(latest),
                _ => None,
            })
        }
        .await;
        match result {
            Ok(Some(latest)) => found.set(format!("↑ v{latest} available: haico --update")),
            Ok(None) => {}
            Err(e) => crate::debug::log("update", &format!("background check failed: {e:#}")),
        }
    });
    available
}

fn client() -> Result<reqwest::Client> {
    // GitHub's API refuses requests without a User-Agent.
    reqwest::Client::builder()
        .user_agent(concat!("Haico/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("cannot set up an HTTP client")
}

pub(crate) async fn run() -> Result<()> {
    let current = env!("CARGO_PKG_VERSION");
    let Some(target) = TARGET else {
        bail!("there are no release builds for this platform; update haico by building it from source");
    };
    let client = client()?;

    println!("Checking for a newer haico than {current}...");
    let release = latest_release(&client, LATEST_RELEASE).await?;
    let (latest, asset) = match check(&release, current, target)? {
        Check::UpToDate => {
            println!("haico {current} is the latest release.");
            return Ok(());
        }
        Check::Ahead(latest) => {
            println!("haico {current} is newer than the latest release ({latest}); nothing to do.");
            return Ok(());
        }
        Check::Update(latest, asset) => (latest, asset),
    };

    println!("Downloading {} ({:.1} MB)...", asset.name, asset.size as f64 / (1024.0 * 1024.0));
    let archive = download(&client, asset).await?;
    let binary = extract_binary(&archive).with_context(|| format!("cannot unpack {}", asset.name))?;
    install(&binary)?;
    println!("Updated haico {current} to {latest}. Release notes: {}", release.html_url);
    Ok(())
}

async fn latest_release(client: &reqwest::Client, url: &str) -> Result<Release> {
    let response = client
        .get(url)
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()
        .await
        .context("cannot reach GitHub to look for a new release")?;
    let status = response.status();
    // 404 is a repository with no release yet; 403 is usually the unauthenticated rate
    // limit (60 requests an hour per address). Either way the status says more than a
    // JSON error about a missing `tag_name` would.
    if !status.is_success() {
        bail!("GitHub answered {status} when asked for the latest release");
    }
    response.json().await.context("cannot read GitHub's answer about the latest release")
}

fn check<'a>(release: &'a Release, current: &str, target: &str) -> Result<Check<'a>> {
    let tag = &release.tag_name;
    let latest = Version::parse(tag.strip_prefix('v').unwrap_or(tag))
        .with_context(|| format!("the latest release's tag '{tag}' is not a version"))?;
    let current = Version::parse(current).context("this haico's own version does not parse")?;
    if latest == current {
        return Ok(Check::UpToDate);
    }
    if latest < current {
        return Ok(Check::Ahead(latest));
    }
    let name = format!("haico-{tag}-{target}.{ARCHIVE}");
    // The workflow publishes the release first and attaches the builds as they finish,
    // so for a few minutes after a tag the release is there and the archive is not.
    let asset = release.assets.iter().find(|a| a.name == name).with_context(|| {
        format!("release {tag} has no {name} yet; if it was published just now, its build may still be running")
    })?;
    Ok(Check::Update(latest, asset))
}

async fn download(client: &reqwest::Client, asset: &Asset) -> Result<Vec<u8>> {
    // The URL redirects to GitHub's object storage; reqwest follows it.
    let bytes = client
        .get(&asset.browser_download_url)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .with_context(|| format!("cannot download {}", asset.name))?
        .bytes()
        .await
        .with_context(|| format!("the download of {} was interrupted", asset.name))?;
    if bytes.len() as u64 != asset.size {
        bail!("downloaded {} bytes of {}, which is {} bytes", bytes.len(), asset.name, asset.size);
    }
    Ok(bytes.to_vec())
}

// Whether an archive entry is the binary, whatever directory it sits in. Both separators:
// the spec says `/`, and an archiver on Windows does not always listen.
fn is_binary(entry: &str) -> bool {
    entry.rsplit(['/', '\\']).next() == Some(BINARY)
}

// Reading an entry to its end checks its CRC, so a corrupted download stops here rather
// than becoming the next haico.
#[cfg(windows)]
fn extract_binary(archive: &[u8]) -> Result<Vec<u8>> {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive))?;
    for index in 0..zip.len() {
        let mut entry = zip.by_index(index)?;
        if entry.is_file() && is_binary(&entry.name()?) {
            let mut binary = Vec::new();
            entry.read_to_end(&mut binary)?;
            return Ok(binary);
        }
    }
    bail!("there is no {BINARY} in it")
}

#[cfg(unix)]
fn extract_binary(archive: &[u8]) -> Result<Vec<u8>> {
    // Decompressed whole before the tar is read: gzip's CRC is checked only at the end of
    // the stream, and the binary comes before the README in the archive, so reading just
    // up to it would never check anything.
    let mut tar_bytes = Vec::new();
    flate2::read::GzDecoder::new(archive).read_to_end(&mut tar_bytes)?;
    let mut tar = tar::Archive::new(tar_bytes.as_slice());
    for entry in tar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_string_lossy().into_owned();
        if entry.header().entry_type().is_file() && is_binary(&path) {
            let mut binary = Vec::new();
            entry.read_to_end(&mut binary)?;
            return Ok(binary);
        }
    }
    bail!("there is no {BINARY} in it")
}

fn install(binary: &[u8]) -> Result<()> {
    // self-replace copies from a file, not from memory. It copies into the binary's own
    // directory before renaming, so where this file lives does not matter.
    let staged = std::env::temp_dir()
        .join(format!("haico-update-{}{}", std::process::id(), std::env::consts::EXE_SUFFIX));
    std::fs::write(&staged, binary)
        .with_context(|| format!("cannot write the new binary to {}", staged.display()))?;
    let replaced = self_replace::self_replace(&staged);
    let _ = std::fs::remove_file(&staged);

    if let Err(e) = replaced {
        let exe = std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_else(|_| "haico".into());
        // The common case: installed somewhere like /usr/local/bin or Program Files.
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            bail!("cannot replace {exe}: permission denied. Run the update with the rights to write there (sudo on Linux, an administrator prompt on Windows).");
        }
        return Err(e).with_context(|| format!("cannot replace {exe}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::serve_once;

    fn release(tag: &str, assets: &[&str]) -> Release {
        Release {
            tag_name: tag.to_string(),
            html_url: format!("https://github.com/Revertron/Haico/releases/tag/{tag}"),
            assets: assets
                .iter()
                .map(|name| Asset { name: name.to_string(), browser_download_url: String::new(), size: 1 })
                .collect(),
        }
    }

    #[test]
    fn a_newer_release_picks_this_targets_archive() {
        let other = format!("haico-v0.4.0-other-target.{ARCHIVE}");
        let mine = format!("haico-v0.4.0-my-target.{ARCHIVE}");
        let release = release("v0.4.0", &[&other, &mine]);
        match check(&release, "0.3.0", "my-target").unwrap() {
            Check::Update(version, asset) => {
                assert_eq!(version, Version::new(0, 4, 0));
                assert_eq!(asset.name, mine);
            }
            other => panic!("expected an update, got {other:?}"),
        }
    }

    #[test]
    fn the_same_or_an_older_release_is_not_an_update() {
        let current = release("v0.3.0", &[]);
        assert!(matches!(check(&current, "0.3.0", "t").unwrap(), Check::UpToDate));
        assert!(matches!(check(&current, "0.3.1", "t").unwrap(), Check::Ahead(_)));
        // Numeric, not textual: "0.10.0" sorts before "0.9.0" as a string.
        let tenth = release("v0.10.0", &[&format!("haico-v0.10.0-t.{ARCHIVE}")]);
        assert!(matches!(check(&tenth, "0.9.0", "t").unwrap(), Check::Update(..)));
    }

    #[test]
    fn a_release_still_building_says_so() {
        let release = release("v0.4.0", &[]);
        let error = check(&release, "0.3.0", "t").unwrap_err().to_string();
        assert!(error.contains("may still be running"), "{error}");
    }

    #[test]
    fn a_tag_that_is_not_a_version_is_refused() {
        let release = release("nightly", &[]);
        let error = check(&release, "0.3.0", "t").unwrap_err().to_string();
        assert!(error.contains("'nightly' is not a version"), "{error}");
    }

    #[tokio::test]
    async fn a_notice_found_before_the_prompt_waits_still_wakes_it() {
        // The check finishes while a turn is running, and the prompt only starts waiting
        // after: the wake has to be kept for it, or the notice sits unseen until a key.
        let available = Available::default();
        assert_eq!(available.notice(), None);
        available.set("news".to_string());
        tokio::time::timeout(Duration::from_secs(1), available.clone().changed())
            .await
            .expect("the wake should have been kept");
        assert_eq!(available.notice().as_deref(), Some("news"));
    }

    #[tokio::test]
    async fn reads_the_release_github_answers_with() {
        let url = serve_once(
            r#"{"tag_name":"v0.4.0","html_url":"https://example/r","draft":false,
                "assets":[{"name":"a.zip","browser_download_url":"https://example/a.zip","size":42}]}"#,
            "application/json",
        )
        .await;
        let release = latest_release(&reqwest::Client::new(), &url).await.unwrap();
        assert_eq!(release.tag_name, "v0.4.0");
        assert_eq!(release.assets[0].size, 42);
    }

    #[test]
    fn finds_the_binary_in_any_directory() {
        assert!(is_binary(&format!("haico-v0.4.0-t/{BINARY}")));
        assert!(is_binary(&format!("haico-v0.4.0-t\\{BINARY}")));
        assert!(is_binary(BINARY));
        assert!(!is_binary("haico-v0.4.0-t/README.md"));
        assert!(!is_binary(&format!("{BINARY}/README.md")));
    }

    // An archive laid out the way the release workflow packs one.
    #[cfg(windows)]
    fn archive(entries: &[(&str, &[u8])]) -> Vec<u8> {
        use std::io::Write;
        use zip::write::SimpleFileOptions;
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for (name, data) in entries {
            zip.start_file(*name, options).unwrap();
            zip.write_all(data).unwrap();
        }
        zip.finish().unwrap().into_inner()
    }

    #[cfg(unix)]
    fn archive(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut tar = tar::Builder::new(Vec::new());
        for (name, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o755);
            header.set_cksum();
            tar.append_data(&mut header, name, *data).unwrap();
        }
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut gz, &tar.into_inner().unwrap()).unwrap();
        gz.finish().unwrap()
    }

    #[test]
    fn unpacks_the_binary_and_nothing_else() {
        let binary_path = format!("haico-v0.4.0-t/{BINARY}");
        let archive = archive(&[("haico-v0.4.0-t/README.md", b"readme"), (&binary_path, b"new haico")]);
        assert_eq!(extract_binary(&archive).unwrap(), b"new haico");
    }

    #[test]
    fn an_archive_without_the_binary_is_refused() {
        let archive = archive(&[("haico-v0.4.0-t/README.md", b"readme")]);
        let error = extract_binary(&archive).unwrap_err().to_string();
        assert!(error.contains("there is no"), "{error}");
    }

    #[test]
    fn a_truncated_archive_is_refused() {
        let binary_path = format!("haico-v0.4.0-t/{BINARY}");
        let archive = archive(&[(&binary_path, &[7u8; 4096])]);
        assert!(extract_binary(&archive[..archive.len() / 2]).is_err());
    }
}
