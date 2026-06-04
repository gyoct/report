#!/usr/bin/env Rscript
# Build index.html in this folder -- a simple landing page that links to every
# per-model report. Edit the LINKS table below to add/rename entries; missing
# files are shown greyed-out as "(not found)" so nothing silently breaks.
#
# Run:  Rscript make_index.R

DIR <- "/home/guanyang/work/report"

# instrument | section | label | file (relative to DIR)
LINKS <- read.table(text = "
btc | linear | Equally weighted | btc_eqweights.html
btc | linear | Weighted         | btc_weighted.html
eth | linear | Equally weighted | eth_eqweights.html
eth | linear | Weighted         | eth_weighted.html
", sep = "|", strip.white = TRUE, stringsAsFactors = FALSE,
   col.names = c("instrument", "section", "label", "file"))

ts_now <- format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")

li <- function(label, file) {
  exists <- file.exists(file.path(DIR, file))
  if (exists) {
    sprintf('<li><a href="%s">%s</a></li>', file, label)
  } else {
    sprintf('<li><span class="missing">%s &mdash; (not found: %s)</span></li>',
            label, file)
  }
}

# Build nested HTML: instrument -> section -> labelled links, in table order.
body <- character(0)
for (inst in unique(LINKS$instrument)) {
  body <- c(body, sprintf("<h2>%s</h2>", toupper(inst)))
  sub <- LINKS[LINKS$instrument == inst, ]
  for (sec in unique(sub$section)) {
    body <- c(body, sprintf("<h3>%s</h3>", sec), "<ol>")
    rows <- sub[sub$section == sec, ]
    for (k in seq_len(nrow(rows))) {
      body <- c(body, li(rows$label[k], rows$file[k]))
    }
    body <- c(body, "</ol>")
  }
}

css <- "
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
     max-width:760px;margin:40px auto;color:#222;line-height:1.5;}
h1{border-bottom:2px solid #444;padding-bottom:6px;}
h2{margin-top:28px;color:#1a3c6e;}
h3{margin:6px 0;color:#555;font-weight:600;}
ol{margin-top:4px;}
li{margin:4px 0;}
a{color:#1558b0;text-decoration:none;} a:hover{text-decoration:underline;}
.missing{color:#999;}
.meta{color:#777;font-size:13px;margin-top:6px;}
"

html <- c(
  "<!DOCTYPE html><html><head><meta charset='utf-8'>",
  "<title>Reports</title>",
  sprintf("<style>%s</style></head><body>", css),
  "<h1>Reports</h1>",
  sprintf("<div class='meta'>generated: %s</div>", ts_now),
  body,
  "</body></html>"
)

out <- file.path(DIR, "index.html")
writeLines(html, out)
cat(sprintf("wrote %s\n", out))
