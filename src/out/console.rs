extern "system" {
    fn GetStdHandle(which: u32) -> isize;
    fn GetConsoleMode(handle: isize, mode: *mut u32) -> i32;
    fn SetConsoleMode(handle: isize, mode: u32) -> i32;
}

const STD_INPUT_HANDLE: u32 = -10i32 as u32;
const ENABLE_VIRTUAL_TERMINAL_INPUT: u32 = 0x0200;
// The three raw mode exists to have off. Named together because they are only ever
// cleared together: echo is documented as having no effect without line input, and
// processed input is what turns Ctrl+C into a signal instead of a byte.
const COOKED: u32 = 0x0001 | 0x0002 | 0x0004;

fn mode() -> Option<u32> {
    unsafe {
        let mut mode = 0;
        if GetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), &mut mode) == 0 {
            return None;
        }
        Some(mode)
    }
}

// Gives back the mode as it was, for putting back later.
pub(super) fn enable_vt_input() -> Option<u32> {
    let original = mode()?;
    unsafe {
        if SetConsoleMode(
            GetStdHandle(STD_INPUT_HANDLE),
            original | ENABLE_VIRTUAL_TERMINAL_INPUT,
        ) == 0
        {
            return None;
        }
    }
    Some(original)
}

pub(super) fn restore(original: u32) {
    unsafe {
        SetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), original);
    }
}

// Put the mode back if something else has changed it, and say nothing when it has not.
//
// The console input mode belongs to the console, not to the process that set it: anything
// sharing this window can rewrite it, and cmd.exe does exactly that — which is why
// `run_command` no longer hands a child the console's own stdin. This is the other half of
// it. Prevention only covers the children haico spawns, and the mode being wrong is not a
// degraded session but an unusable one: cooked input means the console echoes every escape
// sequence into the input box and hands nothing over until Enter, with the transcript on
// the alternate screen where the scrollback cannot be reached either.
//
// Only the flags that matter are forced. Whatever else the mode carries — quick edit
// above all, which is what keeps drag-selection working — is left exactly as found,
// because this runs on every read and is not the place to be making policy.
pub(super) fn reassert_input_mode() {
    let Some(mode) = mode() else {
        return;
    };
    let wanted = (mode & !COOKED) | ENABLE_VIRTUAL_TERMINAL_INPUT;
    if wanted == mode {
        return;
    }
    unsafe {
        SetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), wanted);
    }
}
