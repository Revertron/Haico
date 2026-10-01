//! Timers the model sets on itself: a message that arrives later, as if typed.
//!
//! Three kinds, the three every scheduler ends up with — once after a delay
//! (`setTimeout`), again and again at a fixed period (`setInterval`), and at wall-clock
//! times named by a cron expression. What fires is always the same thing: the text the
//! model gave, handed back to it as a user message that says it came from a timer.
//!
//! This is the store, and only the store. It takes the time as an argument everywhere
//! rather than reading the clock, so the tests can say "ten minutes later" without
//! waiting ten minutes; `tools/timers.rs` is the tool over it and supplies the clock.
//! Delivery is not here either, because it happens in two different places for two
//! different reasons: at a round boundary inside a turn (`run_turn`, the same spot typed-
//! ahead text joins, since anywhere sooner would put a user message between a call and its
//! result), and at the idle prompt (the frontend, which owns the wait for input and so is
//! the only thing that can wait for a timer at the same time).
//!
//! Timers live in memory and die with the session — the conversation that knew why a timer
//! was set is the only thing that can make sense of it firing.

use chrono::{DateTime, Datelike, Local, NaiveDate, NaiveDateTime, TimeZone, Timelike};
use std::time::{Duration, Instant};

/// The most timers armed at once. A model that has set this many is looping, not
/// scheduling.
pub(crate) const MAX_TIMERS: usize = 16;

/// The shortest period an interval may have. Every firing is a whole model turn — seconds
/// to minutes on a local model — so anything faster keeps the GPU busy answering its own
/// alarm. Cron's resolution is a minute anyway.
pub(crate) const MIN_INTERVAL: Duration = Duration::from_secs(60);

/// The longest a one-shot or an interval may be. Further out than this is not something a
/// session is likely to still be open for, and a typo of three extra zeros is.
pub(crate) const MAX_DELAY: Duration = Duration::from_secs(7 * 24 * 3600);

/// How often a cron timer is checked against the wall clock, whatever its monotonic
/// deadline says. The deadline is computed when the timer is armed, and a machine that
/// slept through it — or had its clock changed — must still fire at the next wall-clock
/// match rather than an hour late.
pub(crate) const CRON_RECHECK: Duration = Duration::from_secs(30);

/// When a timer fires.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Schedule {
    /// Once, this long after it was set.
    Once(Duration),
    /// Every this long, the first time one period after it was set.
    Every(Duration),
    /// At every local time the expression matches.
    Cron(Cron),
}

impl Schedule {
    /// A few words for the schedule, used in what the model is told and what the user
    /// sees: "once, after 5 min", "every 90 s", "cron '0 9 * * 1-5'".
    pub(crate) fn describe(&self) -> String {
        match self {
            Schedule::Once(after) => format!("once, after {}", human(*after)),
            Schedule::Every(period) => format!("every {}", human(*period)),
            Schedule::Cron(cron) => format!("cron '{}'", cron.source),
        }
    }

    pub(crate) fn recurring(&self) -> bool {
        !matches!(self, Schedule::Once(_))
    }
}

/// One armed timer.
#[derive(Debug, Clone)]
pub(crate) struct Timer {
    pub(crate) id: u32,
    pub(crate) schedule: Schedule,
    pub(crate) prompt: String,
    /// When it next fires, on the monotonic clock.
    pub(crate) due: Instant,
    /// The same moment on the wall clock, which is what it is shown as, and for a cron
    /// timer what it is really decided by (see [`CRON_RECHECK`]).
    pub(crate) due_wall: DateTime<Local>,
    /// How many times it has fired so far.
    pub(crate) fired: u32,
}

/// A timer that has just gone off.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Fired {
    pub(crate) id: u32,
    pub(crate) schedule: Schedule,
    pub(crate) prompt: String,
    /// Which firing this is, counting from one.
    pub(crate) count: u32,
}

#[derive(Debug, Default)]
pub(crate) struct Timers {
    next_id: u32,
    list: Vec<Timer>,
}

impl Timers {
    /// Arm a timer. `Err` is a sentence for the model saying what to change.
    pub(crate) fn add(
        &mut self,
        schedule: Schedule,
        prompt: &str,
        now: Instant,
        wall: DateTime<Local>,
    ) -> Result<&Timer, String> {
        let prompt = prompt.trim();
        if prompt.is_empty() {
            return Err("Nothing to fire: pass the message the timer should send you in 'prompt'.".to_string());
        }
        if self.list.len() >= MAX_TIMERS {
            return Err(format!(
                "{} timers are already armed, which is the limit. Cancel one you no longer \
                 need before setting another.",
                self.list.len()
            ));
        }
        let (due, due_wall) = match &schedule {
            Schedule::Once(after) | Schedule::Every(after) => {
                if schedule.recurring() && *after < MIN_INTERVAL {
                    return Err(format!(
                        "An interval of {} is too short: every firing is a whole turn of \
                         yours. Use at least {} seconds.",
                        human(*after),
                        MIN_INTERVAL.as_secs()
                    ));
                }
                if after.is_zero() {
                    return Err("A delay of 0 seconds fires at once — just do it now.".to_string());
                }
                if *after > MAX_DELAY {
                    return Err(format!(
                        "{} is longer than this session is going to last. The limit is {}.",
                        human(*after),
                        human(MAX_DELAY)
                    ));
                }
                (now + *after, wall + chrono::Duration::from_std(*after).unwrap_or_default())
            }
            Schedule::Cron(cron) => {
                let Some(next) = cron.next_after(wall) else {
                    return Err(format!(
                        "The cron expression '{}' never matches a real date (like the 31st of \
                         a month with 30 days). Check the day and month fields.",
                        cron.source
                    ));
                };
                (now + until(wall, next), next)
            }
        };
        self.next_id += 1;
        self.list.push(Timer {
            id: self.next_id,
            schedule,
            prompt: prompt.to_string(),
            due,
            due_wall,
            fired: 0,
        });
        Ok(self.list.last().unwrap())
    }

    pub(crate) fn cancel(&mut self, id: u32) -> Option<Timer> {
        let index = self.list.iter().position(|t| t.id == id)?;
        Some(self.list.remove(index))
    }

    /// Disarm everything. Ids keep counting, so a "Timer 1" in the old conversation is
    /// never confused with a new one.
    pub(crate) fn clear(&mut self) -> usize {
        std::mem::take(&mut self.list).len()
    }

    pub(crate) fn list(&self) -> &[Timer] {
        &self.list
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// When whoever is waiting should next look, or `None` with nothing armed.
    pub(crate) fn next_wake(&self, now: Instant) -> Option<Instant> {
        let earliest = self.list.iter().map(|t| t.due).min()?;
        let has_cron = self.list.iter().any(|t| matches!(t.schedule, Schedule::Cron(_)));
        Some(if has_cron { earliest.min(now + CRON_RECHECK) } else { earliest })
    }

    /// Take every timer that is due, rescheduling the recurring ones.
    ///
    /// A recurring timer fires **once** however many periods were missed — a laptop that
    /// slept through six five-minute ticks wakes to one reminder, not six — and its next
    /// firing is counted from now rather than from when it should have fired, so a turn
    /// that ran longer than the period does not come back to find it due again already.
    pub(crate) fn take_due(&mut self, now: Instant, wall: DateTime<Local>) -> Vec<Fired> {
        let mut fired = Vec::new();
        self.list.retain_mut(|timer| {
            let due = match timer.schedule {
                // A wall-clock schedule is decided by the wall clock: see CRON_RECHECK.
                Schedule::Cron(_) => timer.due <= now || timer.due_wall <= wall,
                _ => timer.due <= now,
            };
            if !due {
                return true;
            }
            timer.fired += 1;
            fired.push(Fired {
                id: timer.id,
                schedule: timer.schedule.clone(),
                prompt: timer.prompt.clone(),
                count: timer.fired,
            });
            match &timer.schedule {
                Schedule::Once(_) => false,
                Schedule::Every(period) => {
                    timer.due = now + *period;
                    timer.due_wall = wall + chrono::Duration::from_std(*period).unwrap_or_default();
                    true
                }
                Schedule::Cron(cron) => match cron.next_after(wall) {
                    Some(next) => {
                        timer.due = now + until(wall, next);
                        timer.due_wall = next;
                        true
                    }
                    None => false,
                },
            }
        });
        fired
    }
}

/// What the model is sent when timers fire: one message for however many went off
/// together, each headed with where it came from.
///
/// The header is not decoration. Without it the text reads as the user speaking, and the
/// model answers the user about something the user never said.
pub(crate) fn message_for(fired: &[Fired]) -> String {
    fired
        .iter()
        .map(|f| {
            let when = match f.schedule {
                Schedule::Once(_) => format!("{}; it will not fire again", f.schedule.describe()),
                _ => format!("{}, firing #{}", f.schedule.describe(), f.count),
            };
            format!(
                "[Timer {} fired ({}). This is a reminder you set for yourself, not a \
                 message from the user.]\n{}",
                f.id, when, f.prompt
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// A duration the way a person says it: "45 s", "5 min", "1 h 30 min", "2 d".
pub(crate) fn human(duration: Duration) -> String {
    let total = duration.as_secs();
    let (days, hours, minutes, seconds) =
        (total / 86400, total / 3600 % 24, total / 60 % 60, total % 60);
    let mut parts = Vec::new();
    if days > 0 {
        parts.push(format!("{} d", days));
    }
    if hours > 0 {
        parts.push(format!("{} h", hours));
    }
    if minutes > 0 {
        parts.push(format!("{} min", minutes));
    }
    if seconds > 0 || parts.is_empty() {
        parts.push(format!("{} s", seconds));
    }
    parts.join(" ")
}

fn until(from: DateTime<Local>, to: DateTime<Local>) -> Duration {
    (to - from).to_std().unwrap_or_default()
}

/// A five-field cron expression — minute, hour, day of month, month, day of week —
/// matched against local time.
///
/// Written here rather than taken from a crate because the whole of it is a page, and
/// because what a small model writes is not quite cron: it reaches for `@daily`, for month
/// and day names, for Quartz's `?`. All of those are taken. What is refused, in words, is
/// a six- or seven-field expression, since which field is the seconds is a guess.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Cron {
    pub(crate) source: String,
    minutes: u64,
    hours: u32,
    days: u32,
    months: u16,
    weekdays: u8,
    /// Whether the day-of-month and day-of-week fields were restricted. Cron's oldest
    /// quirk: when both are, a day matches if *either* does ("the 1st, and every Monday").
    days_restricted: bool,
    weekdays_restricted: bool,
}

const MONTH_NAMES: &[&str] = &["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
const DAY_NAMES: &[&str] = &["sun", "mon", "tue", "wed", "thu", "fri", "sat"];

impl Cron {
    pub(crate) fn parse(source: &str) -> Result<Cron, String> {
        let source = source.trim();
        let expanded = match source.to_ascii_lowercase().as_str() {
            "@hourly" => "0 * * * *",
            "@daily" | "@midnight" => "0 0 * * *",
            "@weekly" => "0 0 * * 0",
            "@monthly" => "0 0 1 * *",
            "@yearly" | "@annually" => "0 0 1 1 *",
            other if other.starts_with('@') => {
                return Err(format!(
                    "'{}' is not a cron shorthand. Use @hourly, @daily, @weekly, @monthly or \
                     @yearly, or write the five fields.",
                    source
                ))
            }
            _ => source,
        };
        let fields: Vec<&str> = expanded.split_whitespace().collect();
        if fields.len() != 5 {
            return Err(format!(
                "'{}' has {} fields; a cron expression has exactly five: minute hour \
                 day-of-month month day-of-week, like '*/15 9-17 * * mon-fri'. There is no \
                 seconds field — for that, use an interval instead.",
                source,
                fields.len()
            ));
        }
        let field = |index: usize, min: u32, max: u32, names: &[&str], name: &str| {
            parse_field(fields[index], min, max, names)
                .map_err(|why| format!("The {} field '{}' {}.", name, fields[index], why))
        };
        let minutes = field(0, 0, 59, &[], "minute")?;
        let hours = field(1, 0, 23, &[], "hour")?;
        let days = field(2, 1, 31, &[], "day-of-month")?;
        let months = field(3, 1, 12, MONTH_NAMES, "month")?;
        // 0 through 7, where both ends are Sunday — both conventions are in the wild.
        let weekdays = field(4, 0, 7, DAY_NAMES, "day-of-week")?;
        let weekdays = (weekdays | (weekdays >> 7)) & 0x7F;
        Ok(Cron {
            source: source.to_string(),
            minutes,
            hours: hours as u32,
            days: days as u32,
            months: months as u16,
            weekdays: weekdays as u8,
            days_restricted: !is_wildcard(fields[2]),
            weekdays_restricted: !is_wildcard(fields[4]),
        })
    }

    /// The first local time strictly after `after` that the expression matches, or `None`
    /// when it never does.
    pub(crate) fn next_after(&self, after: DateTime<Local>) -> Option<DateTime<Local>> {
        let start = after.naive_local().with_second(0)?.with_nanosecond(0)?;
        let mut from = start + chrono::Duration::minutes(1);
        // Bounded so an expression that never lands on a real date gives up; a candidate
        // that falls in a DST gap is stepped over, which is what cron daemons do too.
        for _ in 0..8 {
            let candidate = self.next_match(from)?;
            match Local.from_local_datetime(&candidate) {
                chrono::LocalResult::Single(time) if time > after => return Some(time),
                chrono::LocalResult::Ambiguous(early, late) => {
                    if early > after {
                        return Some(early);
                    }
                    if late > after {
                        return Some(late);
                    }
                }
                _ => {}
            }
            from = candidate + chrono::Duration::minutes(1);
        }
        None
    }

    /// The first naive time at or after `from` the fields match, searching up to five
    /// years ahead — long enough for a leap day, short enough that "February 30th" ends.
    pub(crate) fn next_match(&self, from: NaiveDateTime) -> Option<NaiveDateTime> {
        let limit = from.year() + 5;
        let mut t = from;
        while t.year() <= limit {
            if self.months & (1 << t.month()) == 0 {
                let (year, month) = if t.month() == 12 { (t.year() + 1, 1) } else { (t.year(), t.month() + 1) };
                t = NaiveDate::from_ymd_opt(year, month, 1)?.and_hms_opt(0, 0, 0)?;
                continue;
            }
            if !self.day_matches(t.date()) {
                t = t.date().succ_opt()?.and_hms_opt(0, 0, 0)?;
                continue;
            }
            if self.hours & (1 << t.hour()) == 0 {
                t = t.date().and_hms_opt(t.hour(), 0, 0)? + chrono::Duration::hours(1);
                continue;
            }
            if self.minutes & (1 << t.minute()) == 0 {
                t += chrono::Duration::minutes(1);
                continue;
            }
            return Some(t);
        }
        None
    }

    fn day_matches(&self, date: NaiveDate) -> bool {
        let day = self.days & (1 << date.day()) != 0;
        let weekday = self.weekdays & (1 << date.weekday().num_days_from_sunday()) != 0;
        match (self.days_restricted, self.weekdays_restricted) {
            (true, true) => day || weekday,
            (true, false) => day,
            (false, true) => weekday,
            (false, false) => true,
        }
    }
}

fn is_wildcard(field: &str) -> bool {
    field == "*" || field == "?"
}

/// One field into a bitset of the values it allows. Lists, ranges, steps, names.
fn parse_field(field: &str, min: u32, max: u32, names: &[&str]) -> Result<u64, String> {
    let value = |text: &str| -> Result<u32, String> {
        let lower = text.to_ascii_lowercase();
        if let Some(index) = names.iter().position(|n| lower.starts_with(n)) {
            // Months are named from 1, days of the week from 0.
            return Ok(index as u32 + if min == 1 { 1 } else { 0 });
        }
        let n: u32 = text.parse().map_err(|_| format!("has '{}', which is not a number", text))?;
        if n < min || n > max {
            return Err(format!("has {}, outside {}-{}", n, min, max));
        }
        Ok(n)
    };
    let mut bits = 0u64;
    for item in field.split(',') {
        let (range, step) = match item.split_once('/') {
            Some((range, step)) => {
                let step: u32 = step.parse().map_err(|_| format!("has a step '{}' that is not a number", step))?;
                if step == 0 {
                    return Err("has a step of 0".to_string());
                }
                (range, step)
            }
            None => (item, 1),
        };
        let (low, high) = if is_wildcard(range) {
            (min, max)
        } else if let Some((low, high)) = range.split_once('-') {
            (value(low)?, value(high)?)
        } else {
            let start = value(range)?;
            // `5/15` means from 5 to the end in steps of 15, as Vixie cron reads it.
            (start, if step > 1 { max } else { start })
        };
        if low > high {
            return Err(format!("has the range {}-{}, which runs backwards", low, high));
        }
        let mut n = low;
        while n <= high {
            bits |= 1 << n;
            n += step;
        }
    }
    Ok(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(text: &str) -> NaiveDateTime {
        NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M").unwrap()
    }

    fn next(expr: &str, from: &str) -> String {
        Cron::parse(expr).unwrap().next_match(at(from)).unwrap().format("%Y-%m-%d %H:%M %a").to_string()
    }

    #[test]
    fn cron_fields_take_lists_ranges_steps_and_names() {
        assert_eq!(next("*/15 * * * *", "2026-10-01 10:01"), "2026-10-01 10:15 Thu");
        assert_eq!(next("0 9-17 * * mon-fri", "2026-10-02 18:00"), "2026-10-05 09:00 Mon");
        assert_eq!(next("30 8 * * 1,3,5", "2026-10-01 09:00"), "2026-10-02 08:30 Fri");
        assert_eq!(next("0 0 1 jan *", "2026-10-01 00:00"), "2027-01-01 00:00 Fri");
        assert_eq!(next("5/20 * * * *", "2026-10-01 10:26"), "2026-10-01 10:45 Thu");
        // 7 is Sunday as well as 0.
        assert_eq!(next("0 12 * * 7", "2026-10-01 00:00"), "2026-10-04 12:00 Sun");
        assert_eq!(next("@daily", "2026-10-01 00:01"), "2026-10-02 00:00 Fri");
        // Quartz's `?` reads as a wildcard rather than a refusal.
        assert_eq!(next("0 6 ? * *", "2026-10-01 07:00"), "2026-10-02 06:00 Fri");
        // At or after: the start minute itself counts for next_match.
        assert_eq!(next("0 10 * * *", "2026-10-01 10:00"), "2026-10-01 10:00 Thu");
    }

    #[test]
    fn restricting_both_day_fields_matches_either() {
        // The 13th, or any Friday — the next of which here is Friday the 2nd.
        assert_eq!(next("0 0 13 * fri", "2026-10-01 01:00"), "2026-10-02 00:00 Fri");
        assert_eq!(next("0 0 13 * fri", "2026-10-10 01:00"), "2026-10-13 00:00 Tue");
    }

    #[test]
    fn a_date_that_does_not_exist_is_not_waited_for_forever() {
        assert!(Cron::parse("0 0 30 feb *").unwrap().next_match(at("2026-01-01 00:00")).is_none());
        // A leap day is real, just rare.
        assert_eq!(next("0 0 29 feb *", "2026-01-01 00:00"), "2028-02-29 00:00 Tue");
    }

    #[test]
    fn malformed_expressions_say_what_is_wrong() {
        let refusal = |expr: &str| Cron::parse(expr).unwrap_err();
        assert!(refusal("0 */5 * * * *").contains("exactly five"), "{}", refusal("0 */5 * * * *"));
        assert!(refusal("61 * * * *").contains("outside 0-59"));
        assert!(refusal("* * * * blursday").contains("not a number"));
        assert!(refusal("*/0 * * * *").contains("step of 0"));
        assert!(refusal("0 17-9 * * *").contains("backwards"));
        assert!(refusal("@fortnightly").contains("shorthand"));
    }

    fn wall() -> DateTime<Local> {
        Local.with_ymd_and_hms(2026, 10, 1, 12, 0, 0).earliest().unwrap()
    }

    #[test]
    fn a_one_shot_fires_once_and_is_gone() {
        let mut timers = Timers::default();
        let now = Instant::now();
        let id = timers.add(Schedule::Once(Duration::from_secs(90)), "check the build", now, wall()).unwrap().id;
        assert_eq!(timers.next_wake(now), Some(now + Duration::from_secs(90)));

        assert!(timers.take_due(now + Duration::from_secs(89), wall()).is_empty());
        let fired = timers.take_due(now + Duration::from_secs(90), wall());
        assert_eq!(fired.len(), 1);
        assert_eq!((fired[0].id, fired[0].count, fired[0].prompt.as_str()), (id, 1, "check the build"));
        assert!(timers.is_empty());
        assert_eq!(timers.next_wake(now), None);
    }

    #[test]
    fn an_interval_fires_once_per_wake_however_late_and_counts_from_then() {
        let mut timers = Timers::default();
        let now = Instant::now();
        let minute = Duration::from_secs(60);
        timers.add(Schedule::Every(5 * minute), "poll CI", now, wall()).unwrap();

        // Asleep for half an hour: six periods missed, one reminder.
        let late = now + 30 * minute;
        let fired = timers.take_due(late, wall());
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].count, 1);
        // Next one is a period from the wake, not already overdue.
        assert!(timers.take_due(late + minute, wall()).is_empty());
        assert_eq!(timers.take_due(late + 5 * minute, wall())[0].count, 2);
        assert_eq!(timers.list().len(), 1, "an interval stays armed");
    }

    #[test]
    fn a_cron_timer_follows_the_wall_clock_not_only_its_deadline() {
        let mut timers = Timers::default();
        let now = Instant::now();
        let cron = Cron::parse("0 * * * *").unwrap();
        let timer = timers.add(Schedule::Cron(cron), "hourly report", now, wall()).unwrap();
        assert_eq!(timer.due_wall, wall() + chrono::Duration::hours(1));

        // Checked every CRON_RECHECK at most, so a missed deadline is noticed.
        assert_eq!(timers.next_wake(now), Some(now + CRON_RECHECK));
        // The monotonic clock barely moved (the machine slept), the wall clock did.
        let woke = wall() + chrono::Duration::minutes(61);
        let fired = timers.take_due(now + Duration::from_secs(1), woke);
        assert_eq!(fired.len(), 1);
        assert_eq!(timers.list()[0].due_wall, wall() + chrono::Duration::hours(2));
    }

    #[test]
    fn arming_refuses_what_would_misbehave() {
        let mut timers = Timers::default();
        let now = Instant::now();
        let add = |timers: &mut Timers, schedule, prompt: &str| timers.add(schedule, prompt, now, wall()).map(|t| t.id);

        assert!(add(&mut timers, Schedule::Every(Duration::from_secs(10)), "x").unwrap_err().contains("too short"));
        assert!(add(&mut timers, Schedule::Once(Duration::ZERO), "x").unwrap_err().contains("0 seconds"));
        assert!(add(&mut timers, Schedule::Once(MAX_DELAY * 2), "x").unwrap_err().contains("limit"));
        assert!(add(&mut timers, Schedule::Once(Duration::from_secs(5)), "  ").unwrap_err().contains("prompt"));
        let never = Cron::parse("0 0 31 4 *").unwrap();
        assert!(add(&mut timers, Schedule::Cron(never), "x").unwrap_err().contains("never matches"));

        // A one-shot may be short; only repeating has a floor.
        assert_eq!(add(&mut timers, Schedule::Once(Duration::from_secs(5)), "x").unwrap(), 1);
        for _ in 1..MAX_TIMERS {
            add(&mut timers, Schedule::Once(Duration::from_secs(5)), "x").unwrap();
        }
        assert!(add(&mut timers, Schedule::Once(Duration::from_secs(5)), "x").unwrap_err().contains("limit"));

        // Ids are never reused, even across a clear.
        assert_eq!(timers.clear(), MAX_TIMERS);
        assert_eq!(add(&mut timers, Schedule::Once(Duration::from_secs(5)), "x").unwrap(), MAX_TIMERS as u32 + 1);
    }

    #[test]
    fn the_model_is_told_the_message_came_from_its_own_timer() {
        let fired = [
            Fired { id: 2, schedule: Schedule::Once(Duration::from_secs(300)), prompt: "run the tests".into(), count: 1 },
            Fired { id: 3, schedule: Schedule::Every(Duration::from_secs(600)), prompt: "check CI".into(), count: 4 },
        ];
        let message = message_for(&fired);
        assert!(message.contains("[Timer 2 fired (once, after 5 min; it will not fire again)."), "{message}");
        assert!(message.contains("[Timer 3 fired (every 10 min, firing #4)."), "{message}");
        assert!(message.contains("not a message from the user"));
        assert!(message.ends_with("check CI"));
    }

    #[test]
    fn durations_read_like_speech() {
        assert_eq!(human(Duration::from_secs(45)), "45 s");
        assert_eq!(human(Duration::from_secs(300)), "5 min");
        assert_eq!(human(Duration::from_secs(5400)), "1 h 30 min");
        assert_eq!(human(Duration::from_secs(2 * 86400)), "2 d");
        assert_eq!(human(Duration::ZERO), "0 s");
    }
}
