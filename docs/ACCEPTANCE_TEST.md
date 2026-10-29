# Manual acceptance test (about 10 minutes, on your PC)

Run these in order in a copy of your project folder if you are cautious. Back up `outputs\` first:
`xcopy outputs outputs_backup /E /I`.

1. `python -m unittest discover -s tests` -> must end with `OK`. (This writes only to temporary folders.)✅
2. `streamlit run app.py`. Tab "Video and manual log": the video shows and the frame/slider/time boxes agree.✅
3. Press `+1 s`, `-1 s`, `+10 s`, `-10 s`, `+1 frame`, `-1 frame`: the frame, slider and time box change together.✅
   Press `Play 5 s`: frames advance, then the position is kept.✅
4. Go to frame 2384, label **Partial**, notes "acceptance test", keep "Also save evidence frame" ticked, press
   **Add event**. You must see a green "Saved event #N ... to ...events.json" message, the table must list it, and
   an evidence image for that event must appear below the table.✅
5. Go to frame 150, press **Add event** again. The table must now show BOTH events with different ids.✅
6. Stop Streamlit (Ctrl-C in the terminal) and start it again. Both events must still be listed.✅
7. Open `outputs\events.json` and `outputs\events.csv`: both contain the two events with the same ids, frames and
   times. Press **Export CSV + JSON**: the message must say both files hold the same events.✅
8. Select one event, open "Edit this event", change its label, Save changes: the table updates; restart: still there.
9. Delete the frame-150 event with the confirmation tick. It disappears; `outputs\backups\` has `events_predelete_...`;
   its image is in `outputs\deleted_evidence\`. Add a new event: its id must NOT be the deleted id.✅
10. Tab "Review candidates" (after `python tools\make_candidates.py data\video_006.avi`): your earlier `Save test`
    review must still show as confirmed. Save a new decision, restart, confirm it persists, press the export button
    and check `outputs\reviewed\reviewed_events.csv` has `human_` columns and unreviewed rows too.
11. Re-run `make_candidates` with `--gap 2`. The app must either keep your reviews or show a yellow panel listing
    reviews it can re-attach (press the button) or could not match. No review may be attached without that panel.
12. Tab "Experimental": it must say "Not assessable" while `config\calibration.json` does not exist.
13. Failure check: close the app, make `outputs\events.json` unreadable (add a stray `{` at the end of a COPY of the
    project), start the app: it must show an error, refuse to add events, and leave the file unchanged.
14. Browser tab check: open the app in two browser tabs, add an event in each: both must appear in the log.

Report any step that does not behave as described, with the message shown or the traceback.
