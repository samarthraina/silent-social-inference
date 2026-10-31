"""Back up a truncated journal, then remove ONLY its incomplete final line."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('journal',type=Path);a=p.parse_args()
    data=a.journal.read_bytes()
    if not data or data.endswith(b'\n'):raise SystemExit('No incomplete tail; nothing changed.')
    tail=data.rsplit(b'\n',1)[-1]
    try:json.loads(tail)
    except (ValueError,UnicodeError):pass
    else:raise SystemExit('Tail is valid JSON; refusing to remove a complete record. Inspect manually.')
    backup=a.journal.with_name(a.journal.name+'.backup-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    shutil.copy2(a.journal,backup)
    good=data[:len(data)-len(tail)]
    # Check every retained line before replacing anything.
    for line in good.splitlines():json.loads(line)
    a.journal.write_bytes(good);print('Backup:',backup,'; removed incomplete bytes:',len(tail))
