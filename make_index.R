#!/usr/bin/env Rscript
# Build index.html in this folder -- a landing page that links to every
# per-model report, grouped by symbol. For each symbol a card shows TRAINING
# reports on top and PRODUCTION reports on the bottom. Edit the LINKS table
# below to add/rename entries; missing files are shown greyed-out as
# "(not found)" so nothing silently breaks.
#
# Run:  Rscript make_index.R

DIR <- "/home/guanyang/work/report"

# symbol | stage | label | file (relative to DIR)
#   stage must be "Training" or "Production" (controls top/bottom placement).
LINKS <- read.table(text = "
btc | Training   | Equally weighted | btc_eqweights.html
btc | Training   | Weighted         | btc_weighted.html
btc | Production | Monetization     | btc_prod.html
btc | Production | Monetization 2   | btc_prod2.html
eth | Training   | Equally weighted | eth_eqweights.html
eth | Training   | Weighted         | eth_weighted.html
eth | Production | Monetization     | eth_prod.html
", sep = "|", strip.white = TRUE, stringsAsFactors = FALSE,
   col.names = c("symbol", "stage", "label", "file"))

ts_now <- format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")

# A single report rendered as a clickable "pill"; missing files are greyed out.
report_pill <- function(label, file) {
  exists <- file.exists(file.path(DIR, file))
  if (exists) {
    sprintf(paste0(
      '<a class="pill" href="%s">',
      '<span class="pill-label">%s</span>',
      '<span class="pill-file">%s</span></a>'),
      file, label, file)
  } else {
    sprintf(paste0(
      '<span class="pill missing">',
      '<span class="pill-label">%s</span>',
      '<span class="pill-file">not found: %s</span></span>'),
      label, file)
  }
}

# One stage block (Training or Production) inside a symbol card.
stage_block <- function(rows, stage, accent) {
  sub <- rows[rows$stage == stage, ]
  pills <- if (nrow(sub) == 0) {
    '<div class="empty">&mdash;</div>'
  } else {
    paste0('<div class="pills">',
           paste(mapply(report_pill, sub$label, sub$file), collapse = ""),
           '</div>')
  }
  sprintf(paste0(
    '<div class="stage stage-%s">',
    '<div class="stage-head">%s<span class="count">%d</span></div>%s</div>'),
    tolower(stage), stage, nrow(sub), pills)
}

# Build one card per symbol: Training block on top, Production block on bottom.
cards <- character(0)
for (sym in unique(LINKS$symbol)) {
  rows <- LINKS[LINKS$symbol == sym, ]
  cards <- c(cards, sprintf(paste0(
    '<div class="card">',
    '<div class="card-head"><span class="sym">%s</span></div>',
    '%s%s',
    '</div>'),
    toupper(sym),
    stage_block(rows, "Training"),
    stage_block(rows, "Production")))
}

css <- "
:root{
  --bg:#0f172a; --panel:#1e293b; --panel2:#27354a; --line:#334155;
  --txt:#e2e8f0; --muted:#94a3b8; --train:#38bdf8; --prod:#34d399;
}
*{box-sizing:border-box;}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
     margin:0;background:radial-gradient(1200px 600px at 20% -10%,#1e293b,#0f172a);
     color:var(--txt);line-height:1.5;min-height:100vh;}
.wrap{max-width:1040px;margin:0 auto;padding:48px 24px 64px;}
header h1{margin:0;font-size:30px;letter-spacing:.3px;}
header .meta{color:var(--muted);font-size:13px;margin-top:8px;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
      gap:24px;margin-top:32px;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;
      overflow:hidden;box-shadow:0 10px 30px rgba(0,0,0,.35);
      transition:transform .15s ease,box-shadow .15s ease;}
.card:hover{transform:translateY(-3px);box-shadow:0 16px 40px rgba(0,0,0,.45);}
.card-head{padding:18px 22px;background:linear-gradient(90deg,#334155,#1e293b);
           border-bottom:1px solid var(--line);}
.sym{font-size:22px;font-weight:700;letter-spacing:1px;}
.stage{padding:16px 22px;}
.stage + .stage{border-top:1px dashed var(--line);}
.stage-head{display:flex;align-items:center;gap:8px;font-size:12px;
            text-transform:uppercase;letter-spacing:1.5px;font-weight:700;
            margin-bottom:12px;}
.stage-training .stage-head{color:var(--train);}
.stage-production .stage-head{color:var(--prod);}
.count{font-size:11px;color:var(--muted);background:var(--panel2);
       border-radius:10px;padding:1px 8px;font-weight:600;letter-spacing:0;}
.pills{display:flex;flex-direction:column;gap:8px;}
.pill{display:flex;flex-direction:column;text-decoration:none;color:var(--txt);
      background:var(--panel2);border:1px solid var(--line);border-radius:10px;
      padding:10px 14px;transition:border-color .15s,background .15s;}
.pill:hover{background:#30425c;border-color:#475569;}
.stage-training .pill:hover{border-color:var(--train);}
.stage-production .pill:hover{border-color:var(--prod);}
.pill-label{font-weight:600;font-size:14px;}
.pill-file{font-size:11px;color:var(--muted);margin-top:2px;
           font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
.pill.missing{opacity:.5;cursor:not-allowed;}
.pill.missing .pill-file{color:#ef9a9a;}
.empty{color:var(--muted);font-size:13px;padding:4px 0;}
footer{color:var(--muted);font-size:12px;margin-top:40px;text-align:center;}
"

html <- c(
  "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
  "<meta name='viewport' content='width=device-width,initial-scale=1'>",
  "<title>Alpha Reports</title>",
  sprintf("<style>%s</style></head><body>", css),
  "<div class='wrap'>",
  "<header><h1>Alpha Reports</h1>",
  sprintf("<div class='meta'>generated: %s</div></header>", ts_now),
  "<div class='grid'>",
  cards,
  "</div>",
  "<footer>Training reports on top &middot; Production reports below &middot; per symbol</footer>",
  "</div>",
  "</body></html>"
)

out <- file.path(DIR, "index.html")
writeLines(html, out)
cat(sprintf("wrote %s\n", out))
