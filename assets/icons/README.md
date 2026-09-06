# Icons

From [Material Design Icons](https://pictogrammers.com/library/mdi/), which is
licensed Apache-2.0. The path data is theirs and unmodified; the only change is
an explicit `fill` on each path.

That fill is not decoration. An MDI icon ships with no `fill` attribute at all,
so it rasterizes black — invisible on the window's dark skin, which is what the
GUI opens with. Lumio rasterizes an SVG as it finds it and has no tinting step,
so the colour has to be in the file. If the window ever offers a light skin,
these need a light-skin counterpart rather than a runtime tint.

| file | MDI name | used for |
| --- | --- | --- |
| `check.svg` | `check` | the model currently in use |
| `memory.svg` | `memory` | a model already loaded on the server |

`memory` is a RAM chip, which is the distinction exactly: picking a model
without it means waiting for the server to load it first.
