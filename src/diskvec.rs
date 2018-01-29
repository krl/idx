use std::{io, mem, ptr};
use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;
use std::cell::UnsafeCell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::fs::{File, OpenOptions};

use memmap::{MmapMut, MmapOptions};
use parking_lot::{Mutex, MutexGuard};

const RANKS: usize = 128;
const LOCKS: usize = 1024;

pub trait Volatile
where
    Self: Copy + PartialEq,
{
    /// A representation of this type used as a placeholder.
    /// Has to be represented by zeroes only.
    const ZEROED: Self;
}

pub(crate) struct DiskVec<T: Volatile> {
    ranks: [UnsafeCell<Option<MmapMut>>; RANKS],
    initialized: AtomicUsize,
    rank_writelock: Mutex<()>,
    writelocks: [Mutex<()>; LOCKS],
    path: Option<PathBuf>,
    len: AtomicUsize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Volatile> Sync for DiskVec<T> {}
unsafe impl<T: Volatile> Send for DiskVec<T> {}

fn min_max(rank: usize) -> (usize, usize) {
    if rank == 0 {
        (0, 0)
    } else {
        (2usize.pow(rank as u32) - 1, 2usize.pow(rank as u32 + 1) - 2)
    }
}

fn rank_ofs(index: usize) -> (usize, usize) {
    let index = index + 1;
    let rank = mem::size_of::<usize>() * 8 - index.leading_zeros() as usize - 1;
    (rank, index - 2usize.pow(rank as u32))
}

/// A mutable reference into the DiskVec, carrying a guard
pub struct MutableReference<'a, T>
where
    T: 'a,
{
    reference: &'a mut T,
    _guard: MutexGuard<'a, ()>,
}

impl<'a, T> Deref for MutableReference<'a, T>
where
    T: 'a,
{
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.reference
    }
}

impl<'a, T: 'a> DerefMut for MutableReference<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.reference
    }
}

impl<T: Volatile> DiskVec<T> {
    /// Construct a new `DiskVec` given a path.
    pub fn new<P: Into<PathBuf>>(path: P) -> io::Result<Self> {
        Self::initialize(Some(path.into()))
    }

    /// Constructs a new in-memory `DiskVec`.
    pub fn anonymous() -> io::Result<Self> {
        Self::initialize(None)
    }

    fn get_rank_file(
        path: &PathBuf,
        rank: usize,
        create: bool,
    ) -> io::Result<Option<File>> {
        let mut rank_path = path.clone();
        rank_path.push(format!("{}", rank));
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create(create)
            .open(rank_path)
        {
            Ok(file) => Ok(Some(file)),
            Err(err) => if err.kind() == io::ErrorKind::NotFound {
                Ok(None)
            } else {
                Err(err)
            },
        }
    }

    fn initialize(path: Option<PathBuf>) -> io::Result<Self> {
        unsafe {
            #[cfg(not(release))]
            {
                let z: T = mem::zeroed();
                assert!(z == T::ZEROED, "Invalid Volatile implementation");
            }

            let mut ranks: [UnsafeCell<Option<MmapMut>>; RANKS] =
                mem::uninitialized();
            for i in 0..RANKS {
                ptr::write(&mut ranks[i], UnsafeCell::new(None))
            }

            let mut writelocks: [Mutex<()>; LOCKS] = mem::uninitialized();
            for i in 0..LOCKS {
                ptr::write(&mut writelocks[i], Mutex::new(()))
            }

            let mut len = 0;
            let mut n_ranks = 0;

            if let Some(ref path) = path {
                // If we have a path, we first check how many ranks are
                // already present
                for rank in 0..RANKS {
                    if let Some(file) = Self::get_rank_file(path, rank, false)?
                    {
                        n_ranks += 1;
                        let mmap = MmapOptions::new().map_mut(&file)?;
                        *ranks[rank].get() = Some(mmap);
                    } else {
                        break;
                    }
                }

                if n_ranks > 0 {
                    let (mut min, mut max) = min_max(n_ranks - 1);
                    // do a binary search to find the last written
                    // element
                    loop {
                        let probe = min + (max - min) / 2;
                        let (rank, ofs) = rank_ofs(probe);

                        let ptr: *const T = mem::transmute(
                            (*ranks[rank].get())
                                .as_ref()
                                .expect("accessing uninitialized rank")
                                .first(),
                        );
                        let ptr = ptr.offset(ofs as isize);
                        if *ptr != T::ZEROED {
                            // found something
                            if min == max {
                                len = min + 1;
                                break;
                            }
                            min = probe + 1;
                        } else {
                            if min == max {
                                len = min;
                                break;
                            }
                            max = probe;
                        }
                    }
                }
            }

            Ok(DiskVec {
                ranks,
                writelocks,
                len: AtomicUsize::new(len),
                initialized: AtomicUsize::new(n_ranks),
                rank_writelock: Mutex::new(()),
                path: path,
                _marker: PhantomData,
            })
        }
    }

    /// Get a reference to the value at index
    pub fn get(&self, idx: usize) -> Option<&T> {
        let (rank, ofs) = rank_ofs(idx);
        if rank < self.initialized.load(Ordering::Relaxed) {
            unsafe {
                let ptr: *const T = mem::transmute(
                    (*self.ranks[rank].get())
                        .as_ref()
                        .expect("accessing uninitialized rank")
                        .first(),
                );
                let ptr = ptr.offset(ofs as isize);
                if *ptr == T::ZEROED {
                    None
                } else {
                    Some(mem::transmute(ptr))
                }
            }
        } else {
            None
        }
    }

    /// Get a mutable reference to the value at index
    pub fn get_mut(&self, idx: usize) -> Option<MutableReference<T>> {
        let (rank, ofs) = rank_ofs(idx);
        if rank < self.initialized.load(Ordering::Relaxed) {
            unsafe {
                let ptr: *mut T = mem::transmute(
                    (*self.ranks[rank].get())
                        .as_ref()
                        .expect("accessing uninitialized rank")
                        .first(),
                );
                let ptr = ptr.offset(ofs as isize);
                if *ptr == T::ZEROED {
                    None
                } else {
                    Some(MutableReference {
                        reference: mem::transmute(ptr),
                        _guard: self.writelocks[idx % LOCKS].lock(),
                    })
                }
            }
        } else {
            None
        }
    }

    /// returns the length of the `DiskVec`
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Pushes an element to the `DiskVec`, returning the new index.
    pub fn push(&self, t: T) -> io::Result<usize> {
        #[cfg(not(release))]
        assert!(t != T::ZEROED, "Cannot insert zeroes!");

        let idx = self.len.fetch_add(1, Ordering::Relaxed);
        let (rank, ofs) = rank_ofs(idx);

        if rank >= self.initialized.load(Ordering::Relaxed) {
            let _rank_writelock = self.rank_writelock.lock();
            // is the rank still too small after aquiring the lock?
            if rank >= self.initialized.load(Ordering::Relaxed) {
                let n_elements = 2usize.pow(rank as u32);
                let size = mem::size_of::<T>() * n_elements;
                let mmap;

                if let Some(ref path) = self.path {
                    let file = Self::get_rank_file(path, rank, true)?.unwrap();
                    file.set_len(size as u64)?;
                    mmap = unsafe { MmapOptions::new().map_mut(&file)? };
                } else {
                    mmap = MmapOptions::new().len(size).map_anon()?;
                }
                unsafe { *self.ranks[rank].get() = Some(mmap) }
                self.initialized.fetch_add(1, Ordering::Relaxed);
            }
        }

        unsafe {
            let ptr: *const T = mem::transmute(
                (*self.ranks[rank].get())
                    .as_ref()
                    .expect("accessing uninitialized rank")
                    .first(),
            );
            let ptr: *const T = ptr.offset(ofs as isize);
            let ptr: &mut T = mem::transmute(ptr);
            ptr::write(ptr, t);
            Ok(idx)
        }
    }
}

#[cfg(test)]
mod test {
    extern crate tempdir;
    use super::*;
    use self::tempdir::TempDir;
    use std::sync::Arc;
    use std::thread;
    const N: usize = 1_000_000;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct CheckSummedUsize {
        val: usize,
        checksum: usize,
    }

    impl CheckSummedUsize {
        fn new(val: usize) -> Self {
            CheckSummedUsize {
                val,
                checksum: val + 1,
            }
        }
    }

    impl Volatile for CheckSummedUsize {
        const ZEROED: Self = CheckSummedUsize {
            val: 0,
            checksum: 0,
        };
    }

    #[test]
    fn simple_diskarray() {
        let tempdir = TempDir::new("diskarray").unwrap();
        let array = DiskVec::new(tempdir.path()).unwrap();

        for i in 0..N {
            assert_eq!(array.push(CheckSummedUsize::new(i)).unwrap(), i);
        }

        for i in 0..N {
            assert_eq!(array.get(i).unwrap(), &CheckSummedUsize::new(i))
        }

        assert_eq!(array.get(N), None)
    }

    #[test]
    fn diskarray_restore() {
        let tempdir = TempDir::new("diskarray").unwrap();

        {
            let array = DiskVec::new(tempdir.path()).unwrap();

            for i in 0..N {
                assert_eq!(array.push(CheckSummedUsize::new(i)).unwrap(), i);
            }
        }

        {
            let array =
                DiskVec::<CheckSummedUsize>::new(tempdir.path()).unwrap();

            for i in 0..N {
                assert_eq!(array.get(i).unwrap(), &CheckSummedUsize::new(i))
            }
        }
    }

    #[test]
    fn diskarray_len() {
        for little_n in 0..100 {
            let tempdir = TempDir::new("diskarray").unwrap();
            {
                let array = DiskVec::new(tempdir.path()).unwrap();

                for i in 0..little_n {
                    assert_eq!(
                        array.push(CheckSummedUsize::new(i)).unwrap(),
                        i
                    );
                }
            }

            {
                let array =
                    DiskVec::<CheckSummedUsize>::new(tempdir.path()).unwrap();

                assert_eq!(array.len(), little_n);
            }
        }
    }

    #[test]
    fn stress() {
        let tempdir = TempDir::new("diskarray").unwrap();

        let array = Arc::new(DiskVec::new(tempdir.path()).unwrap());

        let n_threads = 16;
        let mut handles = vec![];

        for thread in 0..n_threads {
            let array = array.clone();
            handles.push(thread::spawn(move || {
                for i in 0..N {
                    if i % n_threads == thread {
                        array.push(CheckSummedUsize::new(i)).unwrap();
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(array.len(), N);
    }

    #[test]
    fn mutable_access() {
        let tempdir = TempDir::new("diskarray").unwrap();
        let array = Arc::new(DiskVec::new(tempdir.path()).unwrap());

        for i in 0..N {
            assert_eq!(array.push(CheckSummedUsize::new(i)).unwrap(), i);
        }

        let n_threads = 16;
        let mut handles = vec![];

        for _ in 0..n_threads {
            let array = array.clone();
            handles.push(thread::spawn(move || {
                for i in 0..N {
                    let mut old = array.get_mut(i).unwrap();
                    *old = CheckSummedUsize {
                        val: old.val + 1,
                        checksum: old.checksum + 1,
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        for i in 0..N {
            assert_eq!(
                array.get(i).unwrap(),
                &CheckSummedUsize::new(i + n_threads)
            )
        }
    }

    #[test]
    fn mapping() {
        let mappings: Vec<_> = (0..7).map(rank_ofs).collect();
        assert_eq!(
            mappings,
            vec![(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (2, 3)]
        );
    }

    #[test]
    fn min_max_index() {
        let mappings: Vec<_> = (0..5).map(min_max).collect();
        assert_eq!(mappings, vec![(0, 0), (1, 2), (3, 6), (7, 14), (15, 30)]);
    }
}
