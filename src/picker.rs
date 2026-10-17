//! Picking one item from a list with the arrow keys: the model, the provider.

use crate::input::Keys;
use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEventKind};

// The model list as rows of the bottom region, and which of them the selection is on.
//
// This used to clear the terminal and redraw the list at the top of it, by printing the
// escape for a clear through `out!`. The inline renderer hands such a thing to the
// terminal and forgets it; the full-screen renderer keeps every line it is given and
// paints it again whenever that row comes round, so a stored "clear the screen" wiped the
// frame from the middle of itself and took the input box with it. The region is what both
// renderers already have for something that belongs on the screen rather than in the
// transcript — the permission prompt and the plan review are drawn the same way.
fn picker_rows(
    noun: &str,
    models: &[String],
    selected: usize,
    width: usize,
    room: usize,
) -> (Vec<String>, usize) {
    let columns = width.saturating_sub(1);
    // One row goes to the heading; the rest is the list.
    let body = room.saturating_sub(1).max(1);

    // A window around the selection once the list is longer than there is room for, so a
    // machine with thirty models loaded does not need a screen thirty rows tall.
    let start = if models.len() <= body {
        0
    } else {
        selected
            .saturating_sub(body / 2)
            .min(models.len().saturating_sub(body))
    };
    let end = (start + body).min(models.len());

    let mut rows = vec![crate::input::fit(
        &format!("Select a {noun} — Up/Down to move, Enter to confirm, Esc to cancel"),
        columns,
    )];
    for (offset, model) in models[start..end].iter().enumerate() {
        let line = if start + offset == selected {
            format!("> {} <", model)
        } else {
            format!("  {}", model)
        };
        rows.push(crate::input::fit(&line, columns));
    }

    (rows, 1 + selected - start)
}

pub(crate) async fn select_model_interactive(keys: &mut Keys, models: Vec<String>) -> Result<String> {
    if models.is_empty() {
        return Err(anyhow::anyhow!("No models available"));
    }
    select_interactive(keys, "model", models, 0).await
}

// The same list for anything else picked by name — a provider, today. `noun` is what the
// heading and the transcript call the choice; `selected_index` is where the caret starts,
// which for a provider is the one in force, so Enter alone changes nothing.
pub(crate) async fn select_interactive(
    keys: &mut Keys,
    noun: &str,
    models: Vec<String>,
    selected_index: usize,
) -> Result<String> {
    if models.is_empty() {
        return Err(anyhow::anyhow!("Nothing to choose from"));
    }
    let mut selected_index = selected_index.min(models.len() - 1);

    loop {
        let (width, height) = crate::out::terminal_size().unwrap_or((80, 24));
        // Two rows left to the transcript, so the picker never becomes the whole screen.
        let (rows, caret) =
            picker_rows(noun, &models, selected_index, width.max(20), height.saturating_sub(2));
        crate::out::set_region(rows, caret, 0);

        // Read key event
        let Some(event) = keys.next().await else {
            crate::out::clear_region();
            return Err(anyhow::anyhow!("no interactive terminal to pick a {noun}"));
        };
        // Anything that is not a keystroke redraws the list at the top of the loop instead
        // of ending the picker: a window losing focus is not a choice of model.
        let Event::Key(key_event) = event else {
            continue;
        };

        // Only process key press events, ignore key release
        if key_event.kind != KeyEventKind::Press {
            continue;
        }

        match key_event.code {
            KeyCode::Up => {
                if selected_index > 0 {
                    selected_index -= 1;
                }
            }
            KeyCode::Down => {
                if selected_index < models.len() - 1 {
                    selected_index += 1;
                }
            }
            KeyCode::Enter => {
                // Selection confirmed
                break;
            }
            KeyCode::Esc => {
                crate::out::clear_region();
                return Err(anyhow::anyhow!("{} selection cancelled", capitalized(noun)));
            }
            _ => {
                // Ignore other keys
            }
        }
    }

    // The list comes off the screen; what was chosen goes into the transcript, which is
    // where the rest of the session will be looking for it.
    crate::out::clear_region();
    outln!("Selected {}: {}\n", noun, models[selected_index]);

    Ok(models[selected_index].clone())
}

fn capitalized(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

