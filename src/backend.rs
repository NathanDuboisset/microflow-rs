/// Trait for pluggable computation backends.
///
/// A backend controls how the inner computation loops of the inference ops
/// are scheduled.

/// `defer_job` is given a pointer-sized `arg`. The backend **must** ensure
/// all deferred closures have completed before `wait` returns and before the
/// data pointed to by `arg` is dropped or moved.
pub trait Backend {
    /// Schedule a single unit of work.
    fn defer_job(func: fn(usize), arg: usize);

    /// Block until all jobs previously submitted via [`defer_job`] have completed.
    fn wait();

    /// Evaluates a function across a 2D grid, potentially in parallel.
    fn from_fn<T: Send, F, const ROWS: usize, const COLS: usize>(
        func: F,
    ) -> crate::buffer::Buffer2D<T, ROWS, COLS>
    where
        F: Fn(usize, usize) -> T + Sync,
    {
        use core::mem::MaybeUninit;
        use core::sync::atomic::{AtomicUsize, Ordering};

        // Initialize a 2D array of MaybeUninit explicitly
        let mut output: [[MaybeUninit<T>; ROWS]; COLS] =
            core::array::from_fn(|_| core::array::from_fn(|_| MaybeUninit::uninit()));
        let out_ptr = output.as_mut_ptr() as *mut MaybeUninit<T>;

        struct Ctx<'a, F, T> {
            func: &'a F,
            out_ptr: *mut MaybeUninit<T>,
            next_row: AtomicUsize,
        }

        let ctx = Ctx {
            func: &func,
            out_ptr,
            next_row: AtomicUsize::new(0),
        };

        fn worker<F, T, const R: usize, const C: usize>(arg: usize)
        where
            F: Fn(usize, usize) -> T + Sync,
            T: Send,
        {
            let ctx = unsafe { &*(arg as *const Ctx<'_, F, T>) };
            loop {
                let row = ctx.next_row.fetch_add(1, Ordering::Relaxed);
                if row >= R {
                    break;
                }

                // Compute a whole row (need to have bigger job for it to be worth)
                for col in 0..C {
                    let val = (ctx.func)(row, col);
                    // Column-major: idx = col * R + row
                    let idx = col * R + row;
                    unsafe {
                        ctx.out_ptr.add(idx).write(MaybeUninit::new(val));
                    }
                }
            }
        }

        for _ in 0..ROWS {
            Self::defer_job(worker::<F, T, ROWS, COLS>, &ctx as *const _ as usize);
        }
        Self::wait();

        // Convert the initialized array to SMatrix ArrayStorage safely
        unsafe {
            let ptr = &output as *const _ as *const crate::buffer::Buffer2D<T, ROWS, COLS>;
            let buffer = core::ptr::read(ptr);
            // Forget the original array so its (un)initialized contents aren't dropped
            core::mem::forget(output);
            buffer
        }
    }
}

/// The default sequential backend.
///
/// Every job submitted is run immediately on the calling thread.
pub struct SequentialBackend;

impl Backend for SequentialBackend {
    #[inline(always)]
    fn defer_job(func: fn(usize), arg: usize) {
        func(arg);
    }

    #[inline(always)]
    fn wait() {}

    // Opt to run synchronously for minimal overhead.
    #[inline(always)]
    fn from_fn<T: Send, F, const ROWS: usize, const COLS: usize>(
        func: F,
    ) -> crate::buffer::Buffer2D<T, ROWS, COLS>
    where
        F: Fn(usize, usize) -> T + Sync,
    {
        use core::mem::MaybeUninit;
        let mut output: [[MaybeUninit<T>; ROWS]; COLS] =
            core::array::from_fn(|_| core::array::from_fn(|_| MaybeUninit::uninit()));

        for col in 0..COLS {
            for row in 0..ROWS {
                output[col][row] = MaybeUninit::new(func(row, col));
            }
        }

        unsafe {
            let ptr = &output as *const _ as *const crate::buffer::Buffer2D<T, ROWS, COLS>;
            let buffer = core::ptr::read(ptr);
            core::mem::forget(output);
            buffer
        }
    }
}
