//! Readiness waits: poll a descriptor with a short timeout before a blocking
//! read, so the read starts when the bytes land.
//!
//! Every verb crosses processes: benchd writes the worker's stdin, the worker
//! writes the resident's socket, and each reply comes back the same way. A
//! reader blocked in `read` with no timer pending lets its core drop into a
//! deep idle state, and it runs only after that core has woken up again.
//!
//! [`wait_readable`] polls the descriptor with a 1 ms timeout before the
//! caller's ordinary blocking read. With a timer that close pending, the core
//! stays in a shallow idle state and the reader runs as soon as the bytes
//! arrive. The poll only decides WHEN the read starts: the read, its timeout,
//! its bytes and its errors are the caller's own, unchanged. After `hot`
//! without data it stops polling and the read blocks as before, so an idle
//! worker between phases goes quiet.
//!
//! `CUDA_ENGINE_HOT_WAIT=0` turns it off.

use std::os::fd::RawFd;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

#[repr(C)]
struct PollFd {
    fd: std::ffi::c_int,
    events: std::ffi::c_short,
    revents: std::ffi::c_short,
}

const POLLIN: std::ffi::c_short = 0x001;
/// The poll timeout, in milliseconds (the unit `poll(2)` takes).
const TICK_MS: std::ffi::c_int = 1;

extern "C" {
    fn poll(fds: *mut PollFd, nfds: std::ffi::c_ulong, timeout: std::ffi::c_int)
        -> std::ffi::c_int;
}

/// Whether hot waits are on (`CUDA_ENGINE_HOT_WAIT`, default on).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CUDA_ENGINE_HOT_WAIT").map_or(true, |v| v.trim() != "0"))
}

/// Return once `fd` is readable, hung up or in error, or once `hot` has passed
/// without any of those. Never reads; the caller's blocking read follows.
pub fn wait_readable(fd: RawFd, hot: Duration) {
    if !enabled() {
        return;
    }
    let started = Instant::now();
    loop {
        let mut pfd = PollFd {
            fd,
            events: POLLIN,
            revents: 0,
        };
        // SAFETY: `pfd` is one valid, initialised pollfd that outlives the
        // call, and nfds is 1.
        let ready = unsafe { poll(&mut pfd, 1, TICK_MS) };
        // Ready, hung up, in error, or interrupted: the read decides.
        if ready != 0 || started.elapsed() >= hot {
            return;
        }
    }
}

/// A buffered reader that hot-waits on its descriptor whenever its buffer is
/// empty, so the next line is read the moment it arrives.
pub struct HotReader<R: std::io::Read + std::os::fd::AsRawFd> {
    inner: std::io::BufReader<R>,
    hot: Duration,
}

impl<R: std::io::Read + std::os::fd::AsRawFd> HotReader<R> {
    pub fn new(inner: R, hot: Duration) -> Self {
        Self {
            inner: std::io::BufReader::new(inner),
            hot,
        }
    }
}

impl<R: std::io::Read + std::os::fd::AsRawFd> std::io::Read for HotReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        use std::io::BufRead;
        let available = self.fill_buf()?;
        let n = available.len().min(buf.len());
        buf[..n].copy_from_slice(&available[..n]);
        self.consume(n);
        Ok(n)
    }
}

impl<R: std::io::Read + std::os::fd::AsRawFd> std::io::BufRead for HotReader<R> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        if self.inner.buffer().is_empty() {
            wait_readable(self.inner.get_ref().as_raw_fd(), self.hot);
        }
        self.inner.fill_buf()
    }

    fn consume(&mut self, amt: usize) {
        self.inner.consume(amt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, Write};
    use std::os::unix::net::UnixStream;

    #[test]
    fn hot_reader_reads_the_same_lines() {
        let (mut a, b) = UnixStream::pair().unwrap();
        let writer = std::thread::spawn(move || {
            a.write_all(b"one\ntw").unwrap();
            std::thread::sleep(Duration::from_millis(5));
            a.write_all(b"o\nthree\n").unwrap();
        });
        let reader = HotReader::new(b, Duration::from_secs(2));
        let lines: Vec<String> = reader.lines().map(Result::unwrap).collect();
        writer.join().unwrap();
        assert_eq!(lines, ["one", "two", "three"]);
    }

    #[test]
    fn wait_gives_up_after_hot() {
        let (_a, b) = UnixStream::pair().unwrap();
        let started = Instant::now();
        wait_readable(
            std::os::fd::AsRawFd::as_raw_fd(&b),
            Duration::from_millis(20),
        );
        assert!(started.elapsed() >= Duration::from_millis(20));
    }
}
