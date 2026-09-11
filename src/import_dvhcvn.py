"""Build the offline administrative gazetteer from a pinned upstream snapshot."""
import csv
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = '52a2526adb944f9f5bae2155067beb26f4726bc2'
BASE = f'https://raw.githubusercontent.com/daohoangson/dvhcvn/{REVISION}/'
DEST = Path(__file__).resolve().parent.parent / 'resources' / 'dvhcvn'


def main():
    def fetch(path):
        with urllib.request.urlopen(BASE+path,timeout=60) as response:
            return response.read()
    raw=fetch('data/dvhcvn.json')
    data=json.loads(raw)
    license_text=fetch('LICENSE').decode('utf-8')
    entries=[]
    def visit(record,level,parent=''):
        unit=record['type'].replace(' Trung ương','')
        prefix=unit+' '
        if not record['name'].casefold().startswith(prefix.casefold()):
            # Some upstream type fields disagree with the spelled-out unit.
            # Use the explicit name prefix, keeping the source IDs/hierarchy.
            unit=next((u for u in ('Thành phố','Thị xã','Thị trấn','Tỉnh','Huyện','Quận','Phường','Xã')
                       if record['name'].casefold().startswith(u.casefold()+' ')),None)
            if unit is None:
                raise ValueError(f'Unexpected administrative name: {record["name"]}')
            prefix=unit+' '
        code=record[f'level{level}_id']
        entries.append((code,parent,level,unit.lower(),record['name'][len(prefix):]))
        for child in record.get(f'level{level+1}s',[]):
            visit(child,level+1,code)
    for province in data['data']:
        visit(province,1)
    DEST.mkdir(parents=True,exist_ok=True)
    with (DEST/'units.tsv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle,delimiter='\t')
        writer.writerow(['code','parent','level','unit','name'])
        writer.writerows(entries)
    (DEST/'LICENSE.txt').write_text(license_text,encoding='utf-8')
    (DEST/'SOURCE.md').write_text(
        '# Administrative vocabulary source\n\n'
        'Source: https://github.com/daohoangson/dvhcvn\n\n'
        f'Revision: `{REVISION}`\n\nData date: `{data.get("data_date")}`\n\n'
        f'Original data SHA-256: `{hashlib.sha256(raw).hexdigest()}`\n\n'
        'Derived from data/dvhcvn.json: IDs, parent IDs, levels, units and names; '
        'administrative prefixes separated from names. No upstream program code copied. '
        'Upstream GPL-3.0 license retained in LICENSE.txt. '
        'Historical snapshot, not a claim of current administrative boundaries.\n',encoding='utf-8')
    print('Saved',len(entries),'units to',DEST)
    print('Levels:',{level:sum(r[2]==level for r in entries) for level in (1,2,3)})


if __name__=='__main__':
    main()
