[course proj2 page](https://cal-cs180.github.io/fa26/hw/proj2/index.html)

Due: 11:59pm, Tuesday Sep 29, 2026.

## Assets

All provided assets are downloaded into `data/`:

- `data/cameraman.png` — for Parts 1.2/1.3
- `data/taj.jpg` — for Part 2.1 (unsharp masking)
- `data/apple_orange/{apple,orange}.jpeg` — for Part 2.3/2.4 (multires blending demo)
- `data/hybrid_starter/` — Derek/Nutmeg images + provided starter code for Part 2.2:
  - `DerekPicture.jpg`, `nutmeg.jpg`
  - `align_image_code.py` — provided as-is (interactive point-click alignment via `plt.ginput`)
  - `hybrid_image_starter.py` — skeleton; you implement `hybrid_image()`
- `data/self_portrait.jpg` — for Part 1.1 (converted/downscaled from `proj0/photos/P8310019.ORF`)

- `data/stock/` — public-domain / Creative Commons images from Wikimedia Commons, used for the
  remaining hybrid + blend examples (attribution required by license, credited below):
  - `cat_face.jpg` — Augustus Binu, CC BY-SA 3.0
  - `tiger_face.jpg` — Steve Wilson, CC BY 2.0
  - `lemon.jpg` — Evan-Amos, CC0
  - `lime.jpg` — Ivar Leidus, CC BY-SA 4.0 (cropped from "Lime - whole and halved.jpg")
  - `eye.jpg` — Rapidreflex, CC BY-SA 4.0
  - `moon.jpeg` — NASA/JPL/USGS, public domain

Still needed: a blurry photo to sharpen (2.1, already done with the self-portrait).

## Layout

- `data/` — input images
- `src/` — implementation
- `output/` — generated result images
- `img_submission/` — final images organized by section, for the webpage
- `assignment.md` — full assignment text (verbatim, scraped from the course site)
