//! An append-only, on-disk key-value index
//!
//! Only inserts are supported, updates or deletes not.
#![deny(missing_docs)]
extern crate memmap;
extern crate parking_lot;
extern crate seahash;

mod diskvec;

use std::path::PathBuf;
use std::marker::PhantomData;
use std::{io, mem, ptr};
use std::hash::{Hash, Hasher};

use seahash::SeaHasher;
use diskvec::{DiskVec, Volatile};

const PAGE_SIZE: usize = 4096;

#[derive(Copy)]
struct RawPage<K, V> {
    bytes: [u8; PAGE_SIZE],
    _marker: PhantomData<(K, V)>,
}

impl<K, V> Clone for RawPage<K, V> {
    fn clone(&self) -> Self {
        unsafe {
            let mut page: RawPage<K, V> = mem::uninitialized();
            ptr::copy_nonoverlapping(self, &mut page, 1);
            page
        }
    }
}

impl<K: Copy + PartialEq + Hash, V: Copy + Hash> PartialEq for RawPage<K, V> {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..PAGE_SIZE {
            if self.bytes[i] != other.bytes[i] {
                return false;
            }
        }
        return true;
    }
}

impl<K: Copy + PartialEq + Hash, V: Copy + Hash> Volatile for RawPage<K, V> {
    const ZEROED: Self = RawPage {
        bytes: [0u8; PAGE_SIZE],
        _marker: PhantomData,
    };
}

/// An index mapping keys to values
pub struct Idx<K: Copy + PartialEq + Hash, V: Copy + Hash> {
    vec: DiskVec<RawPage<K, V>>,
}

impl<K: Copy + PartialEq + Hash, V: Copy + Hash> RawPage<K, V> {
    fn new() -> Self {
        let mut page: RawPage<K, V> = unsafe { mem::zeroed() };
        debug_assert!(PAGE_SIZE % 8 == 0, "Page size must be a multiple of 8");
        // cannot be all zeroes
        page.bytes[0] = 255;
        page
    }
}

#[repr(C)]
struct Entry<K, V> {
    k: K,
    v: V,
    checksum: u64,
    next: u64,
    next_b: u64,
}

impl<K: Hash, V: Hash> Entry<K, V> {
    fn valid_key_val(&self) -> bool {
        let mut hasher = SeaHasher::new();
        self.k.hash(&mut hasher);
        self.v.hash(&mut hasher);
        hasher.finish() == self.checksum
    }

    fn next(&self) -> Option<u64> {
        if self.next + 1 == self.next_b {
            Some(self.next)
        } else {
            None
        }
    }
}

enum Probe {
    AlreadyThere,
    Redirect(u64),
    Try,
}

enum Try {
    Ok,
    Redirect(u64),
}

enum Get<'a, V>
where
    V: 'a,
{
    Some(&'a V),
    Redirect(u64),
    None,
}

impl<K: Copy + PartialEq + Hash, V: Copy + Hash> RawPage<K, V> {
    fn probe_insert(&self, slot: u64, k: &K) -> Probe {
        unsafe {
            let entry_ptr: *const Entry<K, V> = mem::transmute(self);
            let entry = entry_ptr.offset(slot as isize);

            if (&*entry).valid_key_val() {
                if (&*entry).k == *k {
                    Probe::AlreadyThere
                } else {
                    match (&*entry).next() {
                        Some(next) => Probe::Redirect(next),
                        None => Probe::Try,
                    }
                }
            } else {
                Probe::Try
            }
        }
    }

    fn try_insert(
        &mut self,
        slot: u64,
        vec: &DiskVec<Self>,
        k: K,
        v: V,
    ) -> io::Result<Try> {
        unsafe {
            let entry_ptr: *mut Entry<K, V> = mem::transmute(self);
            let entry = entry_ptr.offset(slot as isize);

            if (&*entry).valid_key_val() {
                match (&*entry).next() {
                    Some(next) => Ok(Try::Redirect(next)),
                    None => {
                        let idx = vec.push(RawPage::new())?;
                        (&mut *entry).next = idx as u64;
                        (&mut *entry).next_b = idx as u64 + 1;
                        Ok(Try::Redirect(idx as u64))
                    }
                }
            } else {
                let mut hasher = SeaHasher::new();
                k.hash(&mut hasher);
                v.hash(&mut hasher);
                let checksum = hasher.finish();
                ptr::write(
                    entry,
                    Entry {
                        k,
                        v,
                        checksum,
                        next: 0,
                        next_b: 0,
                    },
                );
                Ok(Try::Ok)
            }
        }
    }

    pub fn get(&self, slot: u64, k: &K) -> Get<V> {
        unsafe {
            let entry_ptr: *const Entry<K, V> = mem::transmute(self);
            let entry = entry_ptr.offset(slot as isize);

            if (&*entry).valid_key_val() {
                if (&*entry).k == *k {
                    Get::Some(&(*entry).v)
                } else {
                    match (&*entry).next() {
                        Some(next) => Get::Redirect(next),
                        None => Get::None,
                    }
                }
            } else {
                Get::None
            }
        }
    }
}

impl<K: Copy + PartialEq + Hash, V: Copy + Hash> Idx<K, V> {
    /// Construct a new `Idx` given a path
    pub fn new<P: Into<PathBuf>>(path: P) -> io::Result<Self> {
        let vec = DiskVec::new(path)?;
        if vec.len() == 0 {
            assert_eq!(vec.push(RawPage::new())?, 0)
        }
        Ok(Idx { vec })
    }

    #[inline(always)]
    fn entries_per_page() -> u64 {
        PAGE_SIZE as u64 / mem::size_of::<Entry<K, V>>() as u64
    }

    /// Insert a new key-value pair into the index, if the _key_ is already
    /// there, this is a no-op.
    pub fn insert(&self, k: K, v: V) -> io::Result<()> {
        let mut page: u64 = 0;
        let mut hasher = SeaHasher::new();
        k.hash(&mut hasher);
        let keysum = hasher.finish();
        loop {
            let read =
                self.vec.get(page as usize).expect("invalid page reference");
            let slot = keysum.wrapping_mul(page + 1) % Self::entries_per_page();

            match read.probe_insert(slot, &k) {
                Probe::AlreadyThere => return Ok(()),
                Probe::Redirect(to) => page = to,
                Probe::Try => {
                    let mut write = self.vec
                        .get_mut(page as usize)
                        .expect("invalid page reference");
                    match write.try_insert(slot, &self.vec, k, v)? {
                        Try::Ok => return Ok(()),
                        Try::Redirect(to) => page = to,
                    }
                }
            }
        }
    }

    /// Get the value, if any, associated with key
    pub fn get(&self, k: &K) -> Option<&V> {
        let mut page: u64 = 0;
        let mut hasher = SeaHasher::new();
        k.hash(&mut hasher);
        let keysum = hasher.finish();
        loop {
            let read =
                self.vec.get(page as usize).expect("invalid page reference");
            let slot = keysum.wrapping_mul(page + 1) % Self::entries_per_page();

            match read.get(slot, k) {
                Get::Some(v) => return Some(v),
                Get::Redirect(to) => page = to,
                Get::None => return None,
            }
        }
    }
}

#[cfg(test)]
mod test {
    extern crate tempdir;
    use super::*;
    use self::tempdir::TempDir;
    use self::std::sync::Arc;
    use self::std::thread;
    const N: usize = 100_000;

    #[test]
    fn single_thread() {
        let tempdir = TempDir::new("idx").unwrap();
        let idx = Idx::new(tempdir.path()).unwrap();

        for i in 0..N {
            idx.insert(i, i).unwrap()
        }

        for i in 0..N {
            assert_eq!(idx.get(&i).unwrap(), &i)
        }

        assert_eq!(idx.get(&N), None);
    }

    #[test]
    fn restore() {
        let tempdir = TempDir::new("idx").unwrap();
        {
            let idx = Idx::<usize, usize>::new(tempdir.path()).unwrap();

            for i in 0..N {
                idx.insert(i, i).unwrap()
            }
        }

        let idx = Idx::<usize, usize>::new(tempdir.path()).unwrap();

        for i in 0..N {
            assert_eq!(idx.get(&i).unwrap(), &i)
        }

        assert_eq!(idx.get(&N), None);
    }

    #[test]
    fn multithreading() {
        let tempdir = TempDir::new("idx").unwrap();
        let idx = Arc::new(Idx::new(tempdir.path()).unwrap());

        let n_threads = 16;
        let mut handles = vec![];

        for _ in 0..n_threads {
            let idx = idx.clone();
            handles.push(thread::spawn(move || {
                for i in 0..N {
                    idx.insert(i, i).unwrap();
                }
            }))
        }

        for handle in handles {
            handle.join().unwrap();
        }

        for i in 0..N {
            assert_eq!(idx.get(&i).unwrap(), &i)
        }

        assert_eq!(idx.get(&N), None);
    }
}
