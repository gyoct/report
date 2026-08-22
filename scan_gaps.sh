#!/bin/bash
# Scan a feature CSV's mark_ts (col 2, 1s grid in ns) for gaps != 1s.
f="$1"
awk -F, 'NR==1{next}
  $2 !~ /^[0-9]{18,19}$/ {bad++; next}
  {cur=$2+0;
   if(prev>0){d=cur-prev;
     if(d!=1000000000){g++; if(g<=40) printf "  GAP: %d -> %d  (missing %.2f days / %d s)\n", prev, cur, (d/1e9-1)/86400, d/1e9-1}}
   if(first==0)first=cur; last=cur; prev=cur; n++}
  END{printf "  SUMMARY rows=%d first=%d last=%d gaps=%d bad=%d\n", n, first, last, g+0, bad+0}' "$f"
